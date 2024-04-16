import torch
from torch import Tensor
from torch import nn

class GMMActivation(nn.Module):
    """
    Activation to project data into gaussian mixture model
    """
    mapping = {"p": 0, "x": 1, "y": 2, "dx": 3, "dy": 4, "N": 5, "dN": 6}
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
        inp[:, [0]] = torch.clamp(inp[:, [0]], min=-8., max=8.)
        inp[:, 0] = self.sig(inp[:, 0])
        # 2. Gaussian Mixture Model mean mu
        inp[:, 1:3] = self.tanh(inp[:, 1:3]) #+1.0to -1.0 #half did not work
        # 3. Gaussian Mixture Model sigma
        inp[:, 3:5] = self.sig(inp[:, 3:5])*3+0.001
        #todo: missed 5 background

        # 4. Gaussian Mixture Model intensity
        inp[:, 6] = self.sig(inp[:, 6])*2
        # 5. Background
        inp[:, 7] = self.sig(inp[:, 7])*3

        return inp

class GMMActivationV2(nn.Module):
    """
    Activation to project data into gaussian mixture model
    """
    mapping = {"p": 0, "x": 2, "y": 3, "dx": 5, "dy": 6, "N": 1, "dN": 7,"bg":9}
    sigmoid_ch_ix = [0, 1, 5, 6, 7, 8]  # channel indices with respective activation function
    tanh_ch_ix = [2, 3, 4]

    p_ch_ix = [0]  # channel indices of the respective parameters
    sigma_eps = 0.001
    pxyz_sig_ch_ix = slice(5, 9)
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
        # 1 Clamp Gaussian Mixture Model probabilty p ]0,1[
        inp[:, [0]] = torch.clamp(inp[:, [0]], min=-8., max=8.)

        # 2 Bin probability and sigma to [0,1]
        inp[:, self.sigmoid_ch_ix] = self.sig(inp[:, self.sigmoid_ch_ix])
        # 3 Bin position to [-1,1]
        inp[:, self.tanh_ch_ix] = self.tanh(inp[:, self.tanh_ch_ix])

        # 4 Rescale sigma and add offset
        inp[:, self.pxyz_sig_ch_ix] = inp[:, self.pxyz_sig_ch_ix] * 3 + self.sigma_eps

        return inp