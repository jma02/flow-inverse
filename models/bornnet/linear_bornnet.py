from typing import List, Tuple

import torch
import torch.nn as nn

from ..unet.modules import Downsample, Upsample, ResBlockNoTime


class BornNetLinear(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int = 2,
        ch: int = 32,
        ch_mul: List[int] = [1, 2, 2, 2],
        groups: int = 32,
        dropout: float = 0.1,
        output_shape: Tuple[int, int] = (128, 128),
        rank: int = 1024,
        y_ch: int = 1,
    ):
        super().__init__()

        if y_ch != 1:
            raise ValueError(f"BornNetLinear currently supports y_ch=1, got {y_ch}")

        self.in_ch = in_ch
        self.ch = ch
        self.ch_mul = ch_mul
        self.dropout = dropout
        self.groups = groups
        self.output_shape = output_shape
        self.rank = rank
        self.y_ch = y_ch

        self.input_proj = nn.Conv2d(in_ch, self.ch, 3, 1, 1)

        self.down = nn.ModuleList([])
        self.mid = None
        self.up = nn.ModuleList([])
        self.make_paths()

        self.born_input = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=2 * self.ch),
            nn.SiLU(),
            nn.Conv2d(2 * self.ch, 2, 3, 1, 1),
        )

        h, w = self.output_shape
        self.born_output = nn.Sequential(
            nn.Linear(2 * h * w, rank, bias=False),
            nn.Linear(rank, y_ch * h * w, bias=False),
        )

        self.resnet = nn.ModuleList(
            [
                ResBlockNoTime(1, self.ch, num_groups=1, dropout=self.dropout),
                ResBlockNoTime(self.ch, self.ch, self.groups, self.dropout),
                ResBlockNoTime(self.ch, self.ch, self.groups, self.dropout),
            ]
        )

        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=self.groups, num_channels=self.ch),
            nn.SiLU(),
            nn.Conv2d(self.ch, y_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = nn.AdaptiveAvgPool2d(self.output_shape)(x)

        initial_proj = self.input_proj(x)
        h = initial_proj

        down_path = []
        for block_group in self.down:
            h = block_group[0](h)
            h = block_group[1](h)
            down_path.append(h)

            if len(block_group) > 2:
                h = block_group[2](h)

        h = self.mid[0](h)
        h = self.mid[1](h)

        for block_group in self.up:
            h = torch.cat((h, down_path.pop()), dim=1)
            h = block_group[0](h)
            h = block_group[1](h)

            if len(block_group) > 2:
                h = block_group[2](h)

        h = torch.cat((h, initial_proj), dim=1)
        born_input = self.born_input(h)

        b, c, h_out, w_out = born_input.shape
        born_output_flat = self.born_output(born_input.reshape(b, -1))
        born_output = born_output_flat.reshape(b, self.y_ch, h_out, w_out)

        res = born_output
        for resblock in self.resnet:
            res = resblock(res)

        final_output = self.final(res)
        return final_output, born_output

    def make_transition(self, res: int, down: bool) -> nn.Module:
        dim = self.ch * self.ch_mul[res]

        if down:
            is_last_res = res == (len(self.ch_mul) - 1)
            if is_last_res:
                return Downsample(dim, dim)

            dim_out = self.ch * self.ch_mul[res + 1]
            return Downsample(dim, dim_out)

        is_first_res = res == 0
        if is_first_res:
            return Upsample(dim, dim)

        dim_out = self.ch * self.ch_mul[res - 1]
        return Upsample(dim, dim_out)

    def make_res(self, res: int, down: bool) -> nn.ModuleList:
        dim = self.ch * self.ch_mul[res]
        transition = self.make_transition(res, down)

        if down:
            dim_in = dim
        else:
            dim_in = 2 * dim

        return nn.ModuleList(
            [
                ResBlockNoTime(dim_in, dim, self.groups, self.dropout),
                ResBlockNoTime(dim, dim, self.groups, self.dropout),
                transition,
            ]
        )

    def make_paths(self) -> None:
        num_res = len(self.ch_mul)

        for res in range(num_res):
            is_last_res = res == (num_res - 1)

            down_blocks = self.make_res(res, down=True)
            up_blocks = self.make_res(res, down=False)

            if is_last_res:
                down_blocks = down_blocks[:-1]
            if res == 0:
                up_blocks = up_blocks[:-1]

            self.down.append(down_blocks)
            self.up.insert(0, up_blocks)

        nch = self.ch * self.ch_mul[-1]
        self.mid = nn.ModuleList(
            [
                ResBlockNoTime(nch, nch, self.groups, self.dropout),
                ResBlockNoTime(nch, nch, self.groups, self.dropout),
            ]
        )
