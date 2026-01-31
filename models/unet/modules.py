import math

import numpy as np

import torch
import torch.nn as nn

from ..embeddings import sinusoidal_embedding


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)


    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


def make_skip_connection(dim_in, dim_out):
    if dim_in == dim_out:
        return nn.Identity()
    return nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)

def make_block(dim_in, dim_out, num_groups, dropout=0):
    return nn.Sequential(nn.GroupNorm(num_groups=num_groups, num_channels=dim_in), 
                         nn.SiLU(),
                         nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
                         nn.Conv2d(dim_in, dim_out, 3, 1, 1))


class ConditioningBlock(nn.Module):
    def __init__(self, dim_out, emb_dim):
        super().__init__()
        dim = 2 * dim_out 
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, dim)
        )
    
    def forward(self, emb):
        emb = self.proj(emb)[:, :, None, None]
        return emb
    

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, emb_dim, num_groups=32, dropout=0.1, attn=False):
        super().__init__()

        self.skip_connection = make_skip_connection(dim_in, dim_out)

        self.block1 = make_block(dim_in, dim_out, num_groups, dropout=0)
        self.block2 = make_block(dim_out, dim_out, num_groups, dropout=dropout)
        self.cond_block = ConditioningBlock(dim_out, emb_dim)

    def forward(self, x, emb):
        emb = self.cond_block(emb)

        h = self.block1(x)
        # scale shifting
        out_norm, out_rest = self.block2[0], self.block2[1:]
        scale, shift = emb.chunk(2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)

        h = (self.skip_connection(x) + h) / np.sqrt(2.0)
        return h


class ConditionalResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, emb_dim, num_groups=32, dropout=0.1, attn=False):
        super().__init__()

        self.skip_connection = make_skip_connection(dim_in, dim_out)

        self.block1 = make_block(dim_in, dim_out, num_groups, dropout=0)
        self.block2 = make_block(dim_out, dim_out, num_groups, dropout=dropout)

        self.time_cond = ConditioningBlock(dim_out, emb_dim)
        self.ctx_cond = ConditioningBlock(dim_out, emb_dim)

        nn.init.normal_(self.ctx_cond.proj[1].weight, mean=0.0, std=1e-3)
        if self.ctx_cond.proj[1].bias is not None:
            nn.init.zeros_(self.ctx_cond.proj[1].bias)

    def forward(self, x, time_emb, ctx_emb):
        t = self.time_cond(time_emb)
        c = self.ctx_cond(ctx_emb)
        emb = t + c

        h = self.block1(x)

        out_norm, out_rest = self.block2[0], self.block2[1:]
        scale, shift = emb.chunk(2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)

        h = (self.skip_connection(x) + h) / np.sqrt(2.0)
        return h
    
class ResBlockNoTime(nn.Module):
    def __init__(self, dim_in, dim_out, num_groups=32, dropout=0.1, attn=False):
        super().__init__()

        self.skip_connection = make_skip_connection(dim_in, dim_out)

        self.block1 = make_block(dim_in, dim_out, num_groups, dropout=0)
        self.block2 = make_block(dim_out, dim_out, num_groups, dropout=dropout)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)

        h = (self.skip_connection(x) + h) / np.sqrt(2.0)
        return h


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, downscale_freq_shift: 'float' = 0, max_period: int = 10000):
    return sinusoidal_embedding(
        timesteps,
        embedding_dim,
        downscale_freq_shift=downscale_freq_shift,
        max_period=max_period,
    )