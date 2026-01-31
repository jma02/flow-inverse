from models.unet import UnetNoTime
from models.mlp import MLP

model = UnetNoTime(ch=32)

# print number of parameters
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

model = MLP(in_dim=128, out_dim=128*128, hidden_dims=[64, 100, 90])
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")