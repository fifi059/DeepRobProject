import torch
from network.UNetWithTilingAndFiLM import UNetWithTilingAndFiLM
from network.visual_servoing_networks import SiameseExEy, SiameseEz, SiameseEr

class DomeReimplementation(nn.Module):

  def __init__(self, bottleneck_segmentation, seg_weight_path=None, vsxy_weight_path=None, vsz_weight_path=None, vsr_weight_path=None):
    super(DomeReimplementation, self).__init__()

    self.segmentation_network = UNetWithTilingAndFiLM()
    if seg_weight_path is not None:
      self.segmentation_network.load_state_dict(torch.load(seg_weight_path, weights_only=True))
    self.segmentation_network.eval()
    self.visual_servoing_network_exey = SiameseExEy()
    if vsxy_weight_path is not None:
      self.visual_servoing_network_exey.load_state_dict(torch.load(vsxy_weight_path, weights_only=True))
    self.visual_servoing_network_ez = SiameseEz()
    if vsz_weight_path is not None:
      self.visual_servoing_network_ez.load_state_dict(torch.load(vsz_weight_path, weights_only=True))
    self.visual_servoing_network_er = SiameseEr()
    if vsr_weight_path is not None:
      self.visual_servoing_network_er.load_state_dict(torch.load(vsr_weight_path, weights_only=True))

    self.bottleneck_segmentation = bottleneck_segmentation

    # Freeze the segmentation network
    for param in self.segmentation_network.parameters():
        param.requires_grad = False

    # Switch segmentation network to eval mode
    self.segmentation_network.eval()

  def forward(self, live_image, bottleneck_image):
    segmentation_output = self.segmentation_network(live_image, bottleneck_image)
    live_segmentation = (segmentation_output > 0.5).to(torch.float32) * 255

    batch_size = live_segmentation.size(0)

    ex_ey_output = self.visual_servoing_network_exey(live_segmentation, self.bottleneck_segmentation)
    ex, ey = ex_ey_output[:, 0], ex_ey_output[:, 1]

    ez_output = self.visual_servoing_network_ez(live_segmentation, self.bottleneck_segmentation)
    ez = ez_output[:, 0]

    er_output = self.visual_servoing_network_er(live_segmentation, self.bottleneck_segmentation)
    er_cos, er_sin = er_output[:, 0], er_output[:, 1]

    output_tensor = torch.stack([ex, ey, ez, er_cos, er_sin], dim=1)

    return output_tensor