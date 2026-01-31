
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

class TriangularFineTune(nn.Module):
    def __init__(
        self,
        *,
        y_net: nn.Module,
        y_net_ckpt: str,
        theta_net: nn.Module,
        theta_net_ckpt: str,
        y_ch: int = 1,
        theta_ch: int = 1,
        guidance_scale: float = 1.0,
    ):
        super().__init__()
        assert guidance_scale >= 1.0, ValueError(f"guidance_scale should be >= 1.0, got {guidance_scale}")

        self.y_ch = y_ch
        self.theta_ch = theta_ch
        self.guidance_scale = guidance_scale

        self.y_net = y_net 

        # use ema models for both
        self.theta_net = theta_net
        theta_net_state = torch.load(theta_net_ckpt, map_location="cpu")
        ema_theta = AveragedModel(self.theta_net, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        ema_theta.load_state_dict(theta_net_state["ema_state_dict"])  # handles n_averaged + module.* keys
        self.theta_net.load_state_dict(ema_theta.module.state_dict())     

        self.y_net = y_net
        y_net_state = torch.load(y_net_ckpt, map_location="cpu")
        ema_y = AveragedModel(self.y_net, multi_avg_fn=get_ema_multi_avg_fn(0.999))
        ema_y.load_state_dict(y_net_state["ema_state_dict"])  # handles n_averaged + module.* keys
        self.y_net.load_state_dict(ema_y.module.state_dict())     


    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        assert x_t.ndim == 4, ValueError(f"Expected x_t with shape (B,C,H,W), got {tuple(x_t.shape)}")

        y_t = x_t[:, : self.y_ch]
        theta_t = x_t[:, self.y_ch : self.y_ch + self.theta_ch]

        v_y = self.y_net(y_t, t)

        cfg_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        if cfg_scale == 1.0:
            # we avoid the extra neural field evaluation during training in exchange for a conditional
            v_theta = self.theta_net(theta_t, t)
        else:
            zeros_cond = torch.zeros_like(theta_t)
            v_theta_uncond = self.theta_net(theta_t, zeros_cond, t)
            v_theta_cond = self.theta_net(theta_t, cond_y, t)
            v_theta = v_theta_uncond + cfg_scale * (v_theta_cond - v_theta_uncond)

        return torch.cat([v_y, v_theta], dim=1)