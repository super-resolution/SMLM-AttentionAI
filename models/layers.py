import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from models.util import decoder_patchify, decoder_unpatch


class PositionalEncoding(nn.Module):
    """
    Applies positional encoding to an image
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        """
        Initialization
        :param d_model: embedded dimension of model
        :param dropout: apply dropout to out vector
        :param max_len: Maximal sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)#
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)#positional enconding on sequence i.e. 1
        pe[:, 0::2] = torch.sin(position * div_term[None,:])#1 fold performed worse
        pe[:, 1::2] = torch.cos(position * div_term[None,:-1]) if d_model%2==1 else torch.cos(position * div_term[None,:])
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function of positional encoding
        :param x: Input tensor to apply positional encoding to
        :return: Output tensor with applied dropout and position encoding
        """
        x = x + self.pe[:x.size(0),:,None]
        return self.dropout(x)




class DoubleConv(nn.Module):
    """
    Double Convolution as applied in UNet
    """
    def __init__(self, in_channels:int, out_channels:int, mid_channels=None):
        """
        Initialization
        :param in_channels: Number of input channels
        :param out_channels: Number of desired ouput channels
        :param mid_channels: Number of channels between the convolutions
        """
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

    def forward(self, x:Tensor) -> Tensor:
        """
        Apply double convolution on input tensor
        :param x: Input tensor (image)
        :return: Ouput tensor (convolved image)
        """
        return self.double_conv(x)



class Down(nn.Module):
    """
    Downward convolution as describe in UNet
    """
    def __init__(self, in_channels:int, out_channels:int, pool_size:int):
        """
        Initialization
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param pool_size: Pooling size. Defines how "fast" the size of the image is reduced
        pool_size = 2 halfes the image dimensions
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_size),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x:Tensor) -> Tensor:
        """
        Forward pass
        :param x: Input image as tensor
        :return: Output image
        """
        return self.maxpool_conv(x)



class Up(nn.Module):
    """
    Upward Convolution as described in UNet
    """
    def __init__(self, in_channels:int, out_channels:int, bilinear:bool=True, scale:int=2):
        """
        Initialization
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param bilinear: Use bilinear interpolation to smooth upward convolution
        :param scale: Scaling i.e. magnification for upward convolution
        """
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1:Tensor, x2:Tensor) -> Tensor:
        """
        Forward pass of upward convolution
        :param x1: Input tensor to be upscaled
        :param x2: Concatenation tensor from downward pass
        :return: output tensor
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        #pad stuff if nescesarry
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MHA(nn.Module):
    """
    Implements Multihead Attention
    """
    def __init__(self, embed_dim:int=128, head_dim:int=8, batch_first:bool=False, dropout_rate:float=.1):
        super().__init__()
        #Linear mapping Query q, Key k, and Value v
        self.q = nn.Linear(embed_dim, embed_dim)
        #torch.nn.init.eye_(self.q.weight)
        self.k = nn.Linear(embed_dim, embed_dim)
        #torch.nn.init.eye_(self.k.weight)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        #MHA on first dimension
        self.dropout = nn.Dropout(dropout_rate)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=head_dim, batch_first =batch_first)
    def forward(self, inp:Tensor) -> Tensor:
        """
        Forward pass of multihead attention. Applies residual connection
        :param inp: Input tensor
        :return: Output tensor
        """
        x = inp
        x = self.norm(x)
        #Compute Query Key and Value
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        z,w = self.mha(q,k,v,need_weights=True)
        #w = self.k.weight.cpu().detach().numpy()
        # to plot stuff activate need_weights and selecet z[1]
        # import matplotlib.pyplot as plt
        # plt.bar(list(range(50)),w[12*60+9,0], label="attention")
        # plt.legend()
        z = self.dropout(z)

        # plt.savefig("figures/correlation.svg")
        #plt.show()
        # residual connection + multihead attention
        #Layer norm
        return z + inp

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
    """
    Simple Multilayer perceptron
    Designed in original attention paper
    """
    def __init__(self, embed_dim:int=128, mlp_ratio:int=2):
        """
        Initialization
        :param embed_dim: Used embedded dimension
        :param mlp_ratio: Increase embedded dimension by factor mlp_ratio
        """
        super().__init__()
        self.mlp = nn.Sequential(
        nn.Linear(embed_dim, mlp_ratio * embed_dim),
        nn.GELU(),
        nn.Linear(mlp_ratio * embed_dim, embed_dim)
        )

    def forward(self, inp:Tensor) -> Tensor:
        """
        Forward pass
        :param inp: Input Tensor
        :return: Ouput Tensor
        """
        #Layer norm
        x = self.mlp(inp)
        #residual connection + multihead attention
        return x + inp

class MLP2(nn.Module):
    """
    Multilayer perceptron
    Used in open source diffusion networks. But does not significantly increase the network loss
    Applies residual connection
    """
    def __init__(self, embed_dim:int=128, mlp_ratio:int=2):
        """
        Initialization
        :param embed_dim: Used embedded dimension
        :param mlp_ratio: Increase embedded dimension by factor mlp_ratio
        """
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_dim)#nchannels
        self.lin1 = nn.Linear(embed_dim, mlp_ratio * embed_dim)
        self.lin2 = nn.Linear(embed_dim, mlp_ratio * embed_dim)
        self.act = nn.GELU()
        self.lin3 = nn.Linear(mlp_ratio * embed_dim, embed_dim)

    def forward(self, inp:Tensor) -> Tensor:
        """
        Forward pass
        :param inp: Input Tensor
        :return: Ouput Tensor
        """
        residual_short = inp
        x = self.layernorm(inp)
        x = self.lin1(x)
        gate = self.lin2(inp)
        x = x * self.act(gate)
        x = self.lin3(x)
        return x+residual_short


#Experimental
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





