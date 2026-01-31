import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.0,
        activation: nn.Module = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        if activation is None:
            activation = nn.SiLU()

        layers: list[nn.Module] = []
        dims = [in_dim] + list(hidden_dims) + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == (len(dims) - 2)
            if not is_last:
                layers.append(activation)
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
