import torch
from models import activations
from models.layers import *
#from third_party.decode.models.unet_param import UNet2d
from models.unet import UNet

class Decoder(nn.Module):

    def __init__(self, cfg, hidden_d=128):
        super(Decoder, self).__init__()
        out_ch = (1,4,4,1)
        #todo: this is x*y in u net
        #todo: mha in unet1
        self.mha = MHABlock(embed_dim=hidden_d)
        #self.ca = CABlock(embed_dim=hidden_d)
        self.final = nn.ModuleList([Head(hidden_d, ch) for ch in out_ch])
        torch.nn.init.constant_(self.final[0].out_conv.bias, -6.)
        torch.nn.init.kaiming_normal_(self.final[0].first[0].weight, mode='fan_in',
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.final[0].out_conv.weight, mode='fan_in',
                                      nonlinearity='linear')
        self.unet = UNet(d_size=(hidden_d,hidden_d*2,hidden_d*4), initial_conv=[1,hidden_d])
        self.unet2 = UNet(d_size=(hidden_d,hidden_d*2,hidden_d*4), initial_conv=[hidden_d,hidden_d])


    def forward(self, inp):
        m = inp.max()
        inp = inp/m+0.001#todo: discard
        x = self.unet(inp)
        x = self.mha(x)
        #apply positional encoding on dim1
        #apply mha over batch
        x = self.unet2(x)
        #x = self.ca(y,x)

        #x = x + res_long1
        heads = [f(x) for f in self.final]
        return torch.cat(heads,dim=1)

class Network(nn.Module):

    def __init__(self, cfg):
        #todo: keep base alive for all tests
        super(Network, self).__init__()#load a config for sizes
        self.decoder = Decoder(cfg.decoder, hidden_d=cfg.hidden_d)#downscaling works try further
        #V4 worked best hiddend 400
        #get and initialize defined activation
        self.apply(self.weight_init)
        self.activation = getattr(activations, cfg.activation)()

    def forward(self, input):
        x = self.decoder(input)
        x = self.activation(x)
        return x

    @staticmethod
    def weight_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


class Head(nn.Module):
    def __init__(self, inch,outch):
        super().__init__()
        self.first = torch.nn.Sequential(nn.Conv2d(inch, outch, kernel_size=3, padding="same"),
                            torch.nn.ReLU(),)
        self.out_conv = nn.Conv2d(outch, outch, kernel_size=1, padding="same")

    def forward(self, x):
        x = self.first(x)
        x = self.out_conv(x)
        return x