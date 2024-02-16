import torch
from torch import Tensor
from torch import nn

class GMMActivation(nn.Module):
    """
    Activation to project data into gaussian mixture model
    """
    def __init__(self):
        super().__init__()
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, inp:Tensor) -> Tensor:
        """
        Forward pass
        :param inp: Input Tensor
        :return: Activated output tensor
        """
        # 1. Gaussian Mixture Model probabilty p
        inp[:, 0] = self.sig(inp[:, 0])
        # 2. Gaussian Mixture Model mean mu
        inp[:, 1:3] = self.tanh(inp[:, 1:3]) #+1.0to -1.0 #half did not work
        # 3. Gaussian Mixture Model sigma
        inp[:, 3:5] = self.sig(inp[:, 3:5])*2
        # 4. Gaussian Mixture Model intensity
        inp[:, 6] = self.sig(inp[:, 6])*2
        # 5. Background
        inp[:, 7] = self.sig(inp[:, 7])*3

        return inp