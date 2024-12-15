import torch
import torch.nn as nn
from network.UNetWithTilingAndFiLM import UNetWithTilingAndFiLM
from network.visual_servoing_networks import SiameseExEy, SiameseEz, SiameseEr

class DomeReimplementation(nn.Module):

  def __init__(self, bottleneck_image, seg_weight_path=None, vsxy_weight_path=None, vsz_weight_path=None, vsr_weight_path=None, freeze_seg=True):
    super(DomeReimplementation, self).__init__()

    self.segmentation_network = UNetWithTilingAndFiLM()
    if seg_weight_path is not None:
      self.segmentation_network.load_state_dict(torch.load(seg_weight_path, weights_only=True))
    if freeze_seg:
      # Freeze the segmentation network
      for param in self.segmentation_network.parameters():
          param.requires_grad = False
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

    # Bottleneck Segmentation
    self.bottleneck_image = bottleneck_image
    self.compute_bottleneck_segmentation()

  def forward(self, live_image):
    segmentation_output = self.segmentation_network(live_image, self.bottleneck_image)
    live_segmentation = (segmentation_output > 0.5).to(torch.float32) * 255

    ex_ey_output = self.visual_servoing_network_exey(live_segmentation, self.bottleneck_segmentation)
    ex, ey = ex_ey_output[:, 0], ex_ey_output[:, 1]

    ez_output = self.visual_servoing_network_ez(live_segmentation, self.bottleneck_segmentation)
    ez = ez_output[:, 0]

    er_output = self.visual_servoing_network_er(live_segmentation, self.bottleneck_segmentation)
    er_cos, er_sin = er_output[:, 0], er_output[:, 1]

    output_tensor = torch.stack([ex, ey, ez, er_cos, er_sin], dim=1)

    return output_tensor, live_segmentation

  def to(self, device):
    self.segmentation_network.to(device)
    self.visual_servoing_network_exey.to(device)
    self.visual_servoing_network_ez.to(device)
    self.visual_servoing_network_er.to(device)
    return super().to(device)

  def load_state_dict(self, state_dict, strict=True):
    super(DomeReimplementation, self).load_state_dict(state_dict, strict=strict)
    self.compute_bottleneck_segmentation()

  def compute_bottleneck_segmentation(self):
    with torch.no_grad():
      self.segmentation_network.to(self.bottleneck_image)
      output = self.segmentation_network(self.bottleneck_image, self.bottleneck_image)
      self.bottleneck_segmentation = (output[0] > 0.5).to(torch.float32) * 255