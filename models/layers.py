import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from models.util import decoder_patchify,decoder_unpatch
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)#
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)#positional enconding on sequence i.e. 1
        pe[:, 0, 0::2] = torch.sin(position * div_term)*10#1 fold performed worse
        pe[:, 0, 1::2] = torch.cos(position * div_term)*10
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0),:]
        return x#self.dropout(x)




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
    def __init__(self, embed_dim=128, head_dim=8, batch_first=False):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        #MHA on first dimension
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=head_dim,dropout=0.1, batch_first =batch_first)
    def forward(self, inp):
        #Layer norm
        x = self.norm(inp)
        #Compute Query Key and Value
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        z = self.mha(q,k,v,need_weights=False)[0]
        #y = y.cpu().detach().numpy()
        #todo: also plot close loc
        #this is attention weight of 1
        # plt.bar(list(range(250)),y[12*60+9,0], label="attention")
        # plt.legend()
        # plt.savefig("figures/correlation.svg")
        # plt.show()
        #residual connection + multihead attention
        return  z+ inp

class CA(nn.Module):
    """
    Implements Cross Attention
    """
    def __init__(self, embed_dim=128, head_dim=8, context_dim=128):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(context_dim, embed_dim)
        self.v = nn.Linear(context_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        #MHA on first dimension
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=head_dim,dropout=0.1, batch_first =False)
    def forward(self, x, y):
        residual_short = x
        #Layer norm
        x = self.norm(x)
        #Compute Query Key and Value
        q = self.q(x)
        k = self.k(y)
        v = self.v(y)
        #residual connection + multihead attention
        return self.out(self.mha(q,k,v,need_weights=False)[0]) + residual_short


class MLP(nn.Module):
    def __init__(self, embed_dim=128, mlp_ratio=2):
        super().__init__()
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

class MLP2(nn.Module):
    def __init__(self, embed_dim=128, mlp_ratio=2):
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_dim)#nchannels
        self.lin1 = nn.Linear(embed_dim, mlp_ratio * embed_dim)
        self.lin2 = nn.Linear(embed_dim, mlp_ratio * embed_dim)
        self.act = nn.GELU()
        self.lin3 = nn.Linear(mlp_ratio * embed_dim, embed_dim)

    def forward(self, inp):
        residual_short = inp
        x = self.layernorm(inp)
        x = self.lin1(x)
        gate = self.lin2(inp)
        x = x * self.act(gate)
        x = self.lin3(x)
        return x+residual_short

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128,context_dim=128, mlp_ratio=2, channels=36):
        #n_channels -> patch size
        super().__init__()
        #todo: replace first number with number of in channels
        self.groupnorm = nn.GroupNorm(channels//4, 288, eps=1e-6)
        self.conv_input = nn.Conv2d(288, 288, kernel_size=1, padding=0)
        self.lin_im_to_patch = nn.Linear(800, embed_dim)
        self.lin_context_to_patch = nn.Linear(800, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.mha = MHA(embed_dim=embed_dim, batch_first=False)
        self.ca = CA(embed_dim=embed_dim, context_dim=context_dim)
        self.mlp = MLP2(embed_dim=embed_dim, mlp_ratio=mlp_ratio)
        self.lin_patch_to_im = nn.Linear(embed_dim, 800)
        self.conv_output = nn.Conv2d(288, 288, kernel_size=1, padding=0)
    def forward(self, x, context):
        #x in -> b,c,h,w
        residual_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        x = decoder_patchify(x, 36)
        context = decoder_patchify(context, 36)
        context = self.lin_context_to_patch(context)
        context = self.norm(context)
        x = self.lin_im_to_patch(x)
        #reshape
        #x = x.transpose(-1, -2)  #(n, hw, x)
        #need this norm to not yield nan
        x = self.norm(x)
        x = self.mha(x)
        x = self.ca(x, context)#input context here#todo: activate and debug
        x = self.mlp(x)
        x = self.norm(x)
        x = self.lin_patch_to_im(x)
        #unpatchify
        #reshape back
        x = decoder_unpatch(x, 10)
        return self.conv_output(x) + residual_long


class DiffusionDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #todo: UNet 2 times down conv + cross attention
        #skip connections
        #todo: down along x,y
        #context is image embedding
        #todo: for unfold unpatchify and patchify again with n channels
        self.patch_size = 10
        self.down = nn.ModuleList([
            Down(8,16),
            CrossAttentionBlock(),
            Down(16,32)])
        self.mid = CrossAttentionBlock()
        self.up = nn.ModuleList([
            Up(32,16),
            CrossAttentionBlock(),#48channels in
            Up(48,8)]#todo: compute concatenat input
        )
        self.final = nn.ModuleList([
            nn.GroupNorm(),
            nn.SiLU(),
            nn.Conv2d(in_ch, 8 ,3)]
        )
    def forward(self, x, context):
        #todo: set context to image embedding
        x = x
        stack = []
        for down in self.d_path:
            if isinstance(down,CrossAttentionBlock):
                #todo: change view
                x = down(x,context)
            elif isinstance(down,Down):

                #todo: change view
                # todo: only need to unfold 3 on patch size
                x = down(x)
            stack.append(x)
        x = self.mid(x, context)
        for up in self.u_path:
            if isinstance(up, CrossAttentionBlock):
                #todo: change view
                x = up(torch.concat([x, stack.pop()], dim=-1),context)
            elif isinstance(up, Up):
                #todo: change view
                x = up(x, stack.pop())
        return self.final(x)




#todo: combine attention with u net

