# -*- coding: utf-8 -*-
"""UNET_DOME_FILM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Knb0xduvv4sm1hUu4As29l0uoEtAZQc_

# **DATA LOADING**
"""

from PIL import Image
import os
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, live_image_paths, bottleneck_image_path, mask_paths, transform=None, mask_transform=None):
        self.live_image_paths = sorted([
            os.path.join(live_image_paths, f) for f in os.listdir(live_image_paths) if f.endswith('.png')
        ])
        self.bottleneck_image_path = bottleneck_image_path  # Single bottleneck image
        self.mask_paths = sorted([
            os.path.join(mask_paths, f) for f in os.listdir(mask_paths) if f.endswith('.png')
        ])
        self.transform = transform
        self.mask_transform = mask_transform

        assert len(self.live_image_paths) == len(self.mask_paths), "Mismatch between live images and masks"

    def __len__(self):
        return len(self.live_image_paths)

    def __getitem__(self, idx):

        live_image = Image.open(self.live_image_paths[idx]).convert('RGB')

        bottleneck_image = Image.open(self.bottleneck_image_path).convert('RGB')

        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.transform:
            live_image = self.transform(live_image)
            bottleneck_image = self.transform(bottleneck_image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return live_image, bottleneck_image, mask

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

mask_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

from google.colab import drive
drive.mount('/content/drive')


# Training dataset
train_dataset = SegmentationDataset(
    live_image_paths='/content/drive/MyDrive/DeepRob Project/Traj120/train/images',
    bottleneck_image_path='/content/drive/MyDrive/DeepRob Project/dataset/bottleneck.png',
    mask_paths='/content/drive/MyDrive/DeepRob Project/Traj120/train/masks',
    transform=transform,
    mask_transform=mask_transform
)

# Testing dataset
test_dataset = SegmentationDataset(
    live_image_paths='/content/drive/MyDrive/DeepRob Project/Traj120/test/images',
    bottleneck_image_path='/content/drive/MyDrive/DeepRob Project/dataset/bottleneck.png',
    mask_paths='/content/drive/MyDrive/DeepRob Project/Traj120/test/masks',
    transform=transform,
    mask_transform=mask_transform
)

from torch.utils.data import DataLoader

# DataLoader for training
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# DataLoader for testing
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4
)

print(f"Number of batches in training DataLoader: {len(train_loader)}")
print(f"Number of batches in testing DataLoader: {len(test_loader)}")

live_images, bottleneck_images, masks = next(iter(train_loader))

print(f"Live images batch shape: {live_images.shape}")         # [batch_size, 3, 64, 64]
print(f"Bottleneck images batch shape: {bottleneck_images.shape}")  # [batch_size, 3, 64, 64]
print(f"Masks batch shape: {masks.shape}")                   # [batch_size, 1, 64, 64]

print(f"Each live image size: {live_images[0].shape}")
print(f"Each mask size: {masks[0].shape}")

"""# **UNET FILM**"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ==============================
# Combined U-Net with FiLM in Decoder
# ==============================
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

"""# **TESTING/TRAINING**"""

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = UNetWithTilingAndFiLM()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = FocalLoss(alpha=0.25, gamma=2)
# criterion = nn.BCEWithLogitsLoss()

import csv
import matplotlib.pyplot as plt
import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Parameters
num_epochs = 500
loss_scale = 1e3
weight_saving_step = 10
weight_save_path = "/content/drive/MyDrive/DeepRob Project/Traj120/weights"
loss_save_path = "/content/drive/MyDrive/DeepRob Project/Traj120/losses"
lr_log_path = "/content/drive/MyDrive/DeepRob Project/Traj120/lr_log.csv"

os.makedirs(weight_save_path, exist_ok=True)
os.makedirs(loss_save_path, exist_ok=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, factor=0.75, patience=3)
current_lr = optimizer.param_groups[0]['lr']


loss_history = []
outputs_to_save = []
masks_to_save = []


with open(lr_log_path, mode='w', newline='') as lr_file:
    lr_writer = csv.writer(lr_file)
    lr_writer.writerow(["Epoch", "Learning Rate"])

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for live_image, bottleneck_image, mask in train_loader:
            live_image = live_image.to(device)
            bottleneck_image = bottleneck_image.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(live_image, bottleneck_image)

            loss = criterion(outputs, mask) * loss_scale
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * live_image.size(0)

            outputs_to_save.append(outputs)
            masks_to_save.append(mask)


        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")


        scheduler.step(epoch_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"Learning Rate changed from {current_lr:.8f} to {new_lr:.8f} at epoch {epoch+1}")
            lr_writer.writerow([epoch + 1, new_lr])
            current_lr = new_lr


        if (epoch + 1) % weight_saving_step == 0:
            weight_file = os.path.join(weight_save_path, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), weight_file)
            print(f"Saved weights at epoch {epoch+1} to {weight_file}")


        if (epoch + 1) % weight_saving_step == 0:
            loss_file = os.path.join(loss_save_path, "loss_history.npy")
            torch.save(torch.tensor(loss_history), loss_file)
            print(f"Saved loss history to {loss_file}")

        outputs_to_save.clear()
        masks_to_save.clear()


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()

import matplotlib.pyplot as plt

def visualize_masks(predictions, ground_truth, title="Predicted vs Actual Mask"):

    binary_mask = (predictions > 0.5).float().squeeze(0).cpu().numpy()
    binary_mask = binary_mask.astype('uint8')
    scaled_predicted_mask = binary_mask * 255

    actual_mask = ground_truth.squeeze(0).cpu().numpy()
    scaled_actual_mask = (actual_mask * 255).astype('uint8')


    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(scaled_predicted_mask, cmap='gray', vmin=0, vmax=255)
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(scaled_actual_mask, cmap='gray', vmin=0, vmax=255)
    plt.title("Actual Mask")
    plt.axis('off')

    plt.suptitle(title)
    plt.show()

def test_model(model, test_loader, criterion, device, threshold=0.5, num_samples=5):

    model.eval()
    test_loss = 0.0
    samples_visualized = 0

    with torch.no_grad():
        for live_image, bottleneck_image, mask in test_loader:
            live_image = live_image.to(device)
            bottleneck_image = bottleneck_image.to(device)
            mask = mask.to(device)

            outputs = model(live_image, bottleneck_image)

            loss = criterion(outputs, mask)
            test_loss += loss.item() * live_image.size(0)

            for i in range(live_image.size(0)):
                if samples_visualized < num_samples:
                    visualize_masks(outputs[i], mask[i], f"Sample {samples_visualized + 1}")
                    samples_visualized += 1

            if samples_visualized >= num_samples:
                break

    test_loss /= len(test_loader.dataset)
    print(f"Testing Loss: {test_loss:.4f}")

test_model(model, test_loader, criterion, device)