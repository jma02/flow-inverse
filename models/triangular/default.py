
from typing import Optional

import torch
import torch.nn as nn

from ..unet import ConditionalConcatUnet, Unet


class UnetTriangular(nn.Module):
    def __init__(
        self,
        *,
        y_unet: Optional[nn.Module] = None,
        theta_unet: Optional[nn.Module] = None,
        unet_ch: int = 32,
        y_ch: int = 1,
        theta_ch: int = 1,
        guidance_scale: float = 1.0,
    ):
        super().__init__()
        if y_ch != 1:
            raise ValueError(f"UnetTriangular currently supports y_ch=1, got {y_ch}")
        if theta_ch != 1:
            raise ValueError(f"UnetTriangular currently supports theta_ch=1, got {theta_ch}")
        if guidance_scale < 1.0:
            raise ValueError(f"guidance_scale should be >= 1.0, got {guidance_scale}")

        self.y_ch = y_ch
        self.theta_ch = theta_ch
        self.guidance_scale = guidance_scale

        self.y_unet = y_unet if y_unet is not None else Unet(in_ch=y_ch, ch=unet_ch)
        self.theta_unet = theta_unet if theta_unet is not None else ConditionalConcatUnet(unet_ch=unet_ch)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError(f"Expected x_t with shape (B,C,H,W), got {tuple(x_t.shape)}")
        if x_t.shape[1] != (self.y_ch + self.theta_ch):
            raise ValueError(
                f"Expected x_t channels={self.y_ch + self.theta_ch}, got {x_t.shape[1]}"
            )
        if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError(f"Expected t with shape (B,), got {tuple(t.shape)}")

        y_t = x_t[:, : self.y_ch]
        theta_t = x_t[:, self.y_ch : self.y_ch + self.theta_ch]

        v_y = self.y_unet(y_t, t)
        cfg_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        if cfg_scale == 1.0:
            # we avoid the extra neural field evaluation during training in exchange for a conditional
            v_theta = self.theta_unet(theta_t, y_t, t)
        else:
            zeros_cond = torch.zeros_like(y_t)
            v_theta_uncond = self.theta_unet(theta_t, zeros_cond, t)
            v_theta_cond = self.theta_unet(theta_t, y_t, t)
            v_theta = v_theta_uncond + cfg_scale * (v_theta_cond - v_theta_uncond)

        return torch.cat([v_y, v_theta], dim=1)