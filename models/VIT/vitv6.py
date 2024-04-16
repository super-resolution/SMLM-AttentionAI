import torch

from models import activations
from models.layers import *
from models.unet import UNet
from third_party.decode.models.unet_param import UNet2d

class Decoder(nn.Module):

    def __init__(self, cfg, hidden_d=128, feature_map_d=8, mlp_ratio=2, patch_size=10):
        super(Decoder, self).__init__()
        out_ch = (1,4,4,1)
        #todo: this is x*y in u net
        self.p = PositionalEncoding(hidden_d*4,max_len=50)
        #todo: mha in unet1
        self.mha = MHA(embed_dim=hidden_d*4,head_dim=8, batch_first=False)
        self.mlp = MLP2(embed_dim=hidden_d*4)
        #todo: apply if it works
        #self.p2 = PositionalEncoding(hidden_d*4,max_len=225)
        self.mha2 = MHA(embed_dim=hidden_d*4,head_dim=8, batch_first=True)
        self.mlp2 = MLP2(embed_dim=hidden_d*4)
        self.final = nn.ModuleList([Head(hidden_d, ch) for ch in out_ch])
        torch.nn.init.constant_(self.final[0].out_conv.bias, -6.)
        torch.nn.init.kaiming_normal_(self.final[0].first[0].weight, mode='fan_in',
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.final[0].out_conv.weight, mode='fan_in',
                                      nonlinearity='linear')
        self.groupnorm = nn.GroupNorm(8, hidden_d*4, eps=1e-6)
        self.conv_input = nn.Conv2d(hidden_d*4, hidden_d*4, kernel_size=1, padding=0)
        self.conv_output = nn.Conv2d(hidden_d*4, hidden_d*4, kernel_size=1, padding=0)

        self.unet = UNet2d(1 , 48, depth=2, pad_convs=True,
                                             initial_features=48,
                                             activation=nn.ReLU(), norm=None, norm_groups=None,
                                             pool_mode='StrideConv', upsample_mode='bilinear',
                                             skip_gn_level=None)
        self.unet2 = UNet2d(48 , 48, depth=2, pad_convs=True,
                                             initial_features=48,
                                             activation=nn.ReLU(), norm=None, norm_groups=None,
                                             pool_mode='StrideConv', upsample_mode='bilinear',
                                             skip_gn_level=None)


    def forward(self, inp):
        inp /= inp.max()+0.001#todo: discard
        x = self.unet(inp)
        #concat isntead of add?
        x,enc_out = self.unet2.forward_parts(x, "encoder")
        x = self.unet2.forward_parts(x, "base")
        res_long = x
        b,c,h,w = x.shape
        x = self.groupnorm(x)
        x = self.conv_input(x)
        x = x.view(b,c,h*w)
        x = self.p(x)

        x = x.transpose(-1,-2)
        #apply positional encoding on dim1
        #apply mha over batch
        x = self.mha(x)
        x = self.mlp(x)
        #apply mha over batch
        x = self.mha2(x)
        x = self.mlp2(x)
        x = x.transpose(-1,-2)
        x = x.view(b,c,h,w)
        x = self.conv_output(x)+res_long
        x = self.unet2.forward_parts(x, "decoder", encoder_out=enc_out)
        heads = [f(x) for f in self.final]
        return torch.cat(heads,dim=1)

class ViT(nn.Module):

    def __init__(self, cfg):
        #todo: keep base alive for all tests
        super(ViT, self).__init__()#load a config for sizes
        self.patch_size = cfg.patch_size
        self.decoder = Decoder(cfg.decoder, hidden_d=48, patch_size=self.patch_size)#downscaling works try further
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
