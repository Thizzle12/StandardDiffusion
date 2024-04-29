from torch import nn, Tensor


class DiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, t: int) -> Tensor:
        pass
