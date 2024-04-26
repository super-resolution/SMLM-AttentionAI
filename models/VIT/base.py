import torch
from models import activations
from models.layers import *

class NetworkBase(nn.Module):
    def __init__(self, cfg, hidden_d):
        super(NetworkBase, self).__init__()
        out_ch = (1,4,4,1)
        self.final = nn.ModuleList([Head(hidden_d, ch) for ch in out_ch])
        torch.nn.init.constant_(self.final[3].first[0].bias, 0)
        torch.nn.init.normal_(self.final[3].first[0].weight, 1, 2)
        torch.nn.init.constant_(self.final[3].out_conv.weight, 1)
        torch.nn.init.constant_(self.final[0].out_conv.bias, -6.)
        torch.nn.init.kaiming_normal_(self.final[0].first[0].weight, mode='fan_in',
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.final[0].out_conv.weight, mode='fan_in',
                                      nonlinearity='linear')
        self.apply(self.weight_init)
        self.activation = getattr(activations, cfg.activation)()


    def norm(self, inp):
        m = inp.mean()
        std = inp.std()
        inp = (inp-m)/std#todo: discard
        return inp

    def forward_last(self, x):
        heads = [f(x) for f in self.final]
        x = torch.cat(heads,dim=1)
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