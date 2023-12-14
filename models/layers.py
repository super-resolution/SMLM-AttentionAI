import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_size),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, scale=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MHA(nn.Module):
    """
    Implements Multihead Attention
    """
    def __init__(self, embed_dim=128, head_dim=8):
        super(MHA, self).__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        #MHA on first dimension
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=head_dim,dropout=0.1, batch_first =False)
    def forward(self, inp):
        #Layer norm
        x = self.norm(inp)
        #Compute Query Key and Value
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        #residual connection + multihead attention
        return self.mha(q,k,v,need_weights=False)[0] + inp

class CA(nn.Module):
    """
    Implements Cross Attention
    """
    def __init__(self, embed_dim=128, head_dim=8):
        super(MHA, self).__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        #MHA on first dimension
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=head_dim,dropout=0.1, batch_first =False)
    def forward(self, inp, y):
        #Layer norm
        x = self.norm(inp)
        #Compute Query Key and Value
        q = self.q(y)
        k = self.k(x)
        v = self.v(x)
        #residual connection + multihead attention
        return self.mha(q,k,v,need_weights=False)[0] + inp


class MLP(nn.Module):
    def __init__(self, embed_dim=128, mlp_ratio=2):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
        nn.Linear(embed_dim, mlp_ratio * embed_dim),
        nn.GELU(),
        nn.Linear(mlp_ratio * embed_dim, embed_dim)
        )

    def forward(self, inp):
        #Layer norm
        x = self.mlp(inp)
        #residual connection + multihead attention
        return x + inp

#todo: combine attention with u net

