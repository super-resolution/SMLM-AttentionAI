import torch
from torch import nn


class GMMActivation(nn.Module):
    def __init__(self,):
        super().__init__()
        #todo: determine multiplier in config
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        # 1. Gaussian Mixture Model probabilty p
        input[:, 0] = self.sig(input[:, 0])
        # 2. Gaussian Mixture Model mean mu
        input[:, 1:3] = self.tanh(input[:, 1:3]) #+1.0to -1.0 #half did not work
        # 3. Gaussian Mixture Model sigma
        input[:, 3:5] = self.sig(input[:, 3:5])*2
        # 4. Gaussian Mixture Model intensity
        input[:, 6] = self.sig(input[:, 6])*2
        # 5. Background
        input[:, 7] = self.sig(input[:, 7])*3

        return input