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
    ):
        super().__init__()

        self.y_ch = y_ch
        self.theta_ch = theta_ch

        self.y_net = y_net 

        # use ema models for both, otherwise we load full weights
        if theta_net_ckpt and y_net_ckpt:
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
    ) -> torch.Tensor:
        assert x_t.ndim == 4, ValueError(f"Expected x_t with shape (B,C,H,W), got {tuple(x_t.shape)}")

        y_t = x_t[:, : self.y_ch]
        theta_t = x_t[:, self.y_ch : self.y_ch + self.theta_ch]

        x_y = self.y_net(y_t, t)

        # theta_net should accept x_y as conditioning
        x_theta = self.theta_net(theta_t, t, x_y)

        return torch.cat([x_y, x_theta], dim=1)