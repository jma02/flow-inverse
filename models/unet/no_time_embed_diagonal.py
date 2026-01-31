from typing import List, Optional, Tuple
 
import torch
import torch.nn as nn
 
from .no_time import UnetNoTime
 
 
class UnetNoTimeEmbedDiag(nn.Module):
    def __init__(
        self,
        *,
        diag_dim: int = 128,
        output_shape: Tuple[int, int] = (128, 128),
        proj_hidden_dim: int = 128,
        unet: Optional[nn.Module] = None,
        ch: int = 32,
        ch_mul: List[int] = [1, 2, 2, 2],
        groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
 
        self.diag_dim = diag_dim
        self.output_shape = output_shape
 
        flat_dim = output_shape[0] * output_shape[1]
        self.proj = nn.Sequential(
            nn.Linear(diag_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, flat_dim),
        )
        self.unet = (
            unet
            if unet is not None
            else UnetNoTime(ch=ch, ch_mul=ch_mul, groups=groups, dropout=dropout, output_shape=output_shape)
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (B, D), got {tuple(x.shape)}")
        if x.shape[1] != self.diag_dim:
            raise ValueError(f"Expected diag dim {self.diag_dim}, got {x.shape[1]}")
 
        h = self.proj(x)
        h = h.view(x.shape[0], 1, self.output_shape[0], self.output_shape[1])
        return self.unet(h)