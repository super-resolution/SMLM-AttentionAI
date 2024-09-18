import torch
from models import activations
from models.layers import *
#from third_party.decode.models.unet_param import UNet2d
from models.unet import DiffusionUNet,UNet, SpatialAttentionUNet2
from models.VIT.base import NetworkBase


class Network(NetworkBase):
    def __init__(self, cfg):
        hidden_d = cfg.hidden_d
        super(Network, self).__init__(cfg, hidden_d)#load a config for sizes
        #V4 worked best hiddend 400
        #get and initialize defined activation
        self.unet = UNet(d_size=(hidden_d,hidden_d*2,), initial_conv=[1,hidden_d])
        self.unet2 = SpatialAttentionUNet2((hidden_d,hidden_d*2), initial_conv=[hidden_d,hidden_d])
        self.apply(self.weight_init)

    def forward(self, inp):
        x = self.norm(inp)
        x = self.unet(x)
        x = self.unet2(x)
        x = self.forward_last(x)
        return self.activation(x)