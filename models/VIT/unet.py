import torch
from models import activations
from models.layers import *
from models.unet import DiffusionUNet,UNet
from models.VIT.base import NetworkBase




class Network(NetworkBase):

    def __init__(self, cfg):
        #todo: keep base alive for all tests
        hidden_d = cfg.hidden_d
        super(Network, self).__init__(cfg, cfg.hidden_d)#load a config for sizes
        self.unet = UNet(d_size=(hidden_d,hidden_d*2,hidden_d*4), initial_conv=[1,hidden_d])
        #V4 worked best hiddend 400
        #get and initialize defined activation
        self.apply(self.weight_init)

    def forward(self, inp):
        x = self.norm(inp)
        x = self.unet(x)
        x = self.forward_last(x)
        return self.activation(x)
