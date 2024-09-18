import torch
from models import activations
from models.layers import *
#from third_party.decode.models.unet_param import UNet2d
from models.VIT.base import NetworkBase
from models.unet import UNet

class Network(NetworkBase):

    def __init__(self, cfg):
        hidden_d = cfg.hidden_d
        super(Network, self).__init__(cfg, hidden_d*2)
        self.mha = MHABlock(embed_dim=hidden_d)
        self.unet = UNet(d_size=(hidden_d,hidden_d*2,hidden_d*4), initial_conv=[1,hidden_d])
        self.unet2 = UNet(d_size=(hidden_d*2,hidden_d*4,hidden_d*8), initial_conv=[hidden_d,hidden_d*2])
        #weight init defined in super
        self.apply(self.weight_init)

    def norm(self, inp):
        m = inp.mean()
        std = inp.std()
        inp = (inp-m)/std#todo: discard
        return inp

    def norm2(self, inp):
        mi = inp.min()
        inp = inp-mi
        ma = inp.max()
        inp = inp/ma#todo: discard
        return inp

    def forward(self, inp):
        x = self.norm(inp)
        x = self.unet(x)
        x = self.mha(x)
        #apply positional encoding on dim1
        #apply mha over batch
        x = self.unet2(x)
        x = self.forward_last(x)
        return self.activation(x)