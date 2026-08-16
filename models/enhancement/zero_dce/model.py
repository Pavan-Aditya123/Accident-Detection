"""
Zero-DCE (Zero-Reference Deep Curve Estimation) Architecture & Loss Functions.

Model Architecture:
- DCE-Net: A 7-layer lightweight CNN with symmetrical skip connections.
- Parameter Output: Predicts 24 curve parameter maps (8 iterations * 3 RGB channels).
- Enhancement Function: Applies higher-order quadratic curves iteratively to enhance image brightness and contrast.

Connection with YOLO26:
Low-light/nighttime traffic feeds severely reduce YOLO26 detection accuracy due to dark regions and lost feature gradients.
Zero-DCE operates as an inline real-time pre-processor: raw CCTV frames are enhanced dynamically by DCE-Net 
before being fed into the YOLO26 accident detection pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCENet(nn.Module):
    """
    DCE-Net (Deep Curve Estimation Network).
    Estimates pixel-wise higher-order curve parameters without requiring paired reference images.
    """
    def __init__(self, number_f: int = 32):
        super(DCENet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        # 7-layer CNN with skip connections
        self.conv1 = nn.Conv2d(3, number_f, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(number_f, number_f, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(number_f, number_f, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(number_f, number_f, kernel_size=3, stride=1, padding=1)
        
        # Symmetrical Skip Connections
        self.conv5 = nn.Conv2d(number_f * 2, number_f, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(number_f * 2, number_f, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(number_f * 2, 24, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def enhance(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Applies 8-iteration curve enhancement function:
        LE_n(x) = LE_{n-1}(x) + A_n(x) * LE_{n-1}(x) * (1 - LE_{n-1}(x))
        """
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(A, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        
        return enhance_image

    def forward(self, x: torch.Tensor):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        A = self.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        
        enhanced_image = self.enhance(x, A)
        return enhanced_image, A


# =====================================================================
# ZERO-DCE LOSS FUNCTIONS
# =====================================================================

class SpatialConsistencyLoss(nn.Module):
    """
    Spatial Consistency Loss (L_spa):
    Encourages spatial coherence by preserving local region gradient differences between input and enhanced images.
    """
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        # 4-directional pooling kernels for spatial neighbors
        kernel_left = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4, 4)

    def forward(self, org: torch.Tensor, enhance: torch.Tensor) -> torch.Tensor:
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_left = F.conv2d(org_pool, self.weight_left.to(org.device), padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right.to(org.device), padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up.to(org.device), padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down.to(org.device), padding=1)

        D_enhance_left = F.conv2d(enhance_pool, self.weight_left.to(enhance.device), padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right.to(enhance.device), padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up.to(enhance.device), padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down.to(enhance.device), padding=1)

        d_left = torch.pow(D_org_left - D_enhance_left, 2)
        d_right = torch.pow(D_org_right - D_enhance_right, 2)
        d_up = torch.pow(D_org_up - D_enhance_up, 2)
        d_down = torch.pow(D_org_down - D_enhance_down, 2)

        return torch.mean(d_left + d_right + d_up + d_down)


class ExposureControlLoss(nn.Module):
    """
    Exposure Control Loss (L_exp):
    Measures the distance between local region average intensity and a target exposure level (E = 0.6).
    """
    def __init__(self, patch_size: int = 16, mean_val: float = 0.6):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, enhance: torch.Tensor) -> torch.Tensor:
        x = torch.mean(enhance, 1, keepdim=True)
        mean = self.pool(x)
        return torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(enhance.device), 2))


class ColorConstancyLoss(nn.Module):
    """
    Color Constancy Loss (L_col):
    Enforces the Gray-World assumption, balancing intensity ratios among RGB channels.
    """
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, enhance: torch.Tensor) -> torch.Tensor:
        mean_rgb = torch.mean(enhance, [2, 3])
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)
        
        return torch.mean(torch.pow(d_rg + d_rb + d_gb, 0.5))


class IlluminationSmoothnessLoss(nn.Module):
    """
    Illumination Smoothness Loss (L_tv):
    Total Variation (TV) loss applied to curve parameters A to enforce spatial smoothness and monotonicity.
    """
    def __init__(self, tv_loss_weight: int = 1):
        super(IlluminationSmoothnessLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        batch_size = A.size()[0]
        h_x = A.size()[2]
        w_x = A.size()[3]
        count_h = (A.size()[2] - 1) * A.size()[3]
        count_w = A.size()[2] * (A.size()[3] - 1)
        
        h_tv = torch.pow((A[:, :, 1:, :] - A[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((A[:, :, :, 1:] - A[:, :, :, :w_x - 1]), 2).sum()
        
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class ZeroDCELoss(nn.Module):
    """
    Combined Zero-DCE Multi-Task Loss:
    L_total = L_spa + L_exp + W_col * L_col + W_tv * L_tv
    """
    def __init__(self, w_spa: float = 1.0, w_exp: float = 10.0, w_col: float = 5.0, w_tv: float = 200.0):
        super(ZeroDCELoss, self).__init__()
        self.loss_spa = SpatialConsistencyLoss()
        self.loss_exp = ExposureControlLoss()
        self.loss_col = ColorConstancyLoss()
        self.loss_tv = IlluminationSmoothnessLoss()
        
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_col = w_col
        self.w_tv = w_tv

    def forward(self, org: torch.Tensor, enhance: torch.Tensor, A: torch.Tensor):
        l_spa = self.w_spa * self.loss_spa(org, enhance)
        l_exp = self.w_exp * self.loss_exp(enhance)
        l_col = self.w_col * self.loss_col(enhance)
        l_tv = self.w_tv * self.loss_tv(A)
        
        total_loss = l_spa + l_exp + l_col + l_tv
        return total_loss, {
            "spatial": l_spa.item(),
            "exposure": l_exp.item(),
            "color": l_col.item(),
            "smoothness": l_tv.item(),
            "total": total_loss.item()
        }
