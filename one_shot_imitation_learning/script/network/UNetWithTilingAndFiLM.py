import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWithTilingAndFiLM(nn.Module):
    def __init__(self):
        super(UNetWithTilingAndFiLM, self).__init__()

        encoder_feature_channels = [64, 128, 256, 512]
        self.encoder = UNetEncoder(feature_channels=encoder_feature_channels)

        decoder_feature_channels = [512, 256, 128, 128]
        self.decoder = UNetDecoderWithFiLM(encoder_feature_channels, decoder_feature_channels)

        conditioning_output_dims = [256] + [2 * size for size in decoder_feature_channels]
        self.conditioning_net = ConditioningNetwork(out_dims=conditioning_output_dims)

        self.post_convs = nn.Sequential(
            nn.Conv2d(decoder_feature_channels[-1], 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),
        )

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, live_image, bottleneck_image):
        conditioning_outputs = self.conditioning_net(bottleneck_image)
        tiling_vector = conditioning_outputs[0]
        film_parameters_list = conditioning_outputs[1:]

        encoder_features = self.encoder(live_image, tiling_vector)
        x = encoder_features[-1]
        x = self.decoder(x, encoder_features, film_parameters_list)

        x = self.post_convs(x)
        # print(f"After post_convs: {x.shape}")

        x = self.final_conv(x)
        # print(f"After final_conv: {x.shape}")

        x = self.sigmoid(x)
        # print(f"Final output shape: {x.shape}\n")

        return x

# ==============================
# Encoder with Tiling Vector
# ==============================
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, tiling_dim=256, feature_channels=[64, 128, 256, 512]):

        super(UNetEncoder, self).__init__()
        self.tiling_dim = tiling_dim
        self.feature_channels = feature_channels

        self.layers = nn.ModuleList()
        current_in_channels = in_channels + tiling_dim

        for out_channels in feature_channels:
            self.layers.append(self._conv_block(current_in_channels, out_channels))
            current_in_channels = out_channels + tiling_dim

    def _conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25)
        )
        return block

    def forward(self, x, tiling_vector):
        features = []
        B, _, H, W = x.shape

        for idx, layer in enumerate(self.layers):
            tiling_expanded = tiling_vector.unsqueeze(-1).unsqueeze(-1).expand(B, self.tiling_dim, H, W)
            x = torch.cat([x, tiling_expanded], dim=1)
            # print(f"Encoder layer {idx+1}:")
            # print(f"  Input shape (after concatenation): {x.shape}")
            x = layer(x)
            # print(f"  Output shape: {x.shape}\n")
            features.append(x)
            H, W = x.shape[2], x.shape[3]

        return features

# ==============================
# Decoder with FiLM
# ==============================
class UNetDecoderWithFiLM(nn.Module):
    def __init__(self, encoder_feature_channels, decoder_feature_channels=[512, 256, 128, 128], out_channels=1):
        super(UNetDecoderWithFiLM, self).__init__()

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.film_blocks = nn.ModuleList()

        self.decoder_feature_channels = decoder_feature_channels

        for idx, out_ch in enumerate(decoder_feature_channels):
            self.upconvs.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
            if idx == 0:
                in_channels = encoder_feature_channels[-1] + encoder_feature_channels[-2]
            if idx>0 and idx!=3:
                in_channels = decoder_feature_channels[idx - 1] + encoder_feature_channels[-(idx + 2)]
            if idx==3 :
                in_channels = 128

            self.dec_blocks.append(
                self._conv_block(in_channels, out_ch)
            )

            self.film_blocks.append(FiLMBlock(out_ch))

        self.final_conv = nn.Conv2d(decoder_feature_channels[-1], out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25)
        )
        return block

    def forward(self, x, encoder_features, film_parameters_list):
        for idx, (upconv, dec_block, film_block) in enumerate(zip(self.upconvs, self.dec_blocks, self.film_blocks)):
            # print(f"Decoder layer {idx + 1}:")
            x = upconv(x)
            # print(f"  After upconv: {x.shape}")

            if idx < len(encoder_features) - 1:
                enc_feat = encoder_features[-(idx + 2)]
                # print(f"  Encoder feature shape: {enc_feat.shape}")
                x = torch.cat([x, enc_feat], dim=1)
                # print(f"  After concatenation: {x.shape}")

            x = dec_block(x)
            # print(f"  After dec_block: {x.shape}")

            gamma, beta = torch.chunk(film_parameters_list[idx], chunks=2, dim=1)
            # print(f"  FiLM gamma shape: {gamma.shape}, beta shape: {beta.shape}")

            x = film_block(x, gamma, beta)
            # print(f"  After FiLMBlock: {x.shape}\n")

        return x

# ==============================
# Conditioning Network
# ==============================
class ConditioningNetwork(nn.Module):
    def __init__(self, out_dims, num_heads=1):

        super(ConditioningNetwork, self).__init__()

        # CNN encoder for bottleneck image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # [B,64,32,32]
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B,128,16,16]
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B,256,8,8]
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # [B,256,4,4]
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),

            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # [B,128,2,2]
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),
        )
        self.flatten = nn.Flatten()
        self.heads = nn.ModuleList([
            self._fc_block(128 * 2 * 2, out_dim) for out_dim in out_dims
        ])

    def _fc_block(self, in_features, out_features):

        block = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),

            nn.Linear(256, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),

            nn.Linear(256, out_features)
        )
        return block

    def forward(self, x):

        x = self.encoder(x)
        # print(f"Conditioning encoder output shape: {x.shape}")

        x = self.flatten(x)
        # print(f"Flattened output shape: {x.shape}")

        outputs = []
        for idx, head in enumerate(self.heads):
            out = head(x)
            # print(f"Head {idx+1} output shape: {out.shape}")
            outputs.append(out)

        return outputs

# ==============================
# FiLM Block
# ==============================
class FiLMBlock(nn.Module):
    def __init__(self, num_features):

        super(FiLMBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(num_features)

    def forward(self, x, gamma, beta):
        # residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        residual = x
        x = self.conv2(x)
        x = self.norm2(x)
        # FiLM modulation
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)
        x = gamma * x + beta
        x = F.relu(x)
        x = x + residual
        return x