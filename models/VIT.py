import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor
import math
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import numpy as np
from models.unet import UNet
from models.layers import *
from models import activations
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
        return self.dropout(x)


def gaussian_initializer(shape=(10,10,40,40)):
    im = torch.zeros((100,40,40))
    for i in range(10):
        for j in range(10):
            im[i*10+j,i*39//9,j*39//9] = 1
    gaussian = lambda x: (torch.exp(-(x) ** 2 / (2 * (2) ** 2)))
    w = gaussian(torch.arange(-10,11,1.0))
    im = F.pad(im, (10,10,10,10), "reflect")
    res = F.conv2d(im[:,None], w[None,None,None,:], padding="valid")
    res = F.conv2d(res, w[None,None,:,None], padding="valid")
    res =  res.flatten(start_dim=1)

    return res


class ImageEmbedding(torch.nn.Module):
    def __init__(self, chw=(100,60,60),patch_size=10, hidden_d=128):
        super(ImageEmbedding, self).__init__()
        # Attributes
        double_conv_width = 8
        self.chw = chw # (C, H, W)
        #assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        #assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = patch_size
        self.n_patches = (chw[1]//self.patch_size,chw[2]//self.patch_size)
        self.hidden_d = hidden_d
        # 1) Linear mapper
        self.input_d = int(self.patch_size**2 * double_conv_width)
        # project on 8 channels with double conv
        self.conv = DoubleConv(1,double_conv_width)
        #takes 8 feauture maps
        self.unet = UNet()
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        #no initial values for linea mapping?



        #self.linear_mapper.weight.data = torch.permute(gaussian_initializer(),(1,0))
        #self.linear_mapper.bias.data.fill_(0.0)
        #layer norm before embedding?
        #discard image normalization and rather increase pos embedding

        # 2) Learnable classification token todo test if this works
        #self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        # 3) Positional embedding
        self.pos_embed = PositionalEncoding(self.hidden_d)

    def patchify(self, images):
        patches = images.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = torch.reshape(patches, (patches.shape[0], patches.shape[1] * patches.shape[2], -1))
        return patches

    def forward(self,images):
        images = self.conv(images[:,None])
        images = self.unet(images)
        #todo: apply residual connection?

        images = torch.permute(images, (0,2,3,1))
        patches = self.patchify(images)
        tokens = self.linear_mapper(patches)
        # Adding classification token to the tokens
        #tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # Adding positional embedding
        # apply positional encoding over batch size
        out = self.pos_embed(tokens)
        return out




class Encoder(nn.Module):
    def __init__(self, cfg, hidden_d=128, mlp_ratio=2, patch_size=10):
        super(Encoder, self).__init__()
        #min U-Net
        #embedding + MHA
        #decoding
        #todo: this could be sequential
        #todo: define in VIT
        self.embedding = ImageEmbedding(hidden_d=hidden_d, patch_size=patch_size)
        self.mha = MHA(embed_dim=hidden_d)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = MLP(embed_dim=hidden_d, mlp_ratio=mlp_ratio)
        #self.mh_attention2 = MHA(embed_dim=hidden_d)
        self.norm3 = nn.LayerNorm(hidden_d)

    def forward(self, input):
        x1 = self.embedding(input)
        x = self.mha(x1)
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.norm3(x + x1)
        #residual connection
        #todo: dont use residual conenctions afert norm!
        return x

class Decoder(nn.Module):
    def __init__(self, cfg, hidden_d=128, feature_map_d=8, mlp_ratio=2, patch_size=10):
        super(Decoder, self).__init__()
        #todo: input dimension by config/patch size

        # self.mh_attention = MHA(embed_dim=hidden_d)
        # self.norm = nn.LayerNorm(hidden_d)
        # self.dec = nn.Sequential(
        #     nn.Linear(hidden_d, mlp_ratio * hidden_d),
        #     nn.GELU(),
        #     nn.Linear(mlp_ratio * hidden_d,hidden_d))
        # self.norm2 = nn.LayerNorm(hidden_d)

        self.patch_size = patch_size
        self.linear = nn.Linear(hidden_d, patch_size**2*feature_map_d)
        # initializor
        # eye = torch.eye(hidden_d)[None,None, ...]
        #
        # def get_scale_mat(m):
        #     scale_mat = torch.tensor([[1, 0., 0.],
        #                               [0., m, 0.]])
        #     return scale_mat
        # mat = get_scale_mat(patch_size[0]*patch_size[1]*feature_map_d//hidden_d)[None, ...]
        # grid = F.affine_grid(mat, eye.size())
        # eye = F.grid_sample(eye, grid)
        # self.linear.weight.data = torch.eye(patch_size[0]*patch_size[1]*feature_map_d, hidden_d)
        # self.linear.bias.data.fill_(0.0)
        self.norm3 = nn.LayerNorm(patch_size**2 * feature_map_d)

    def forward(self, input, h):
        #todo: map larger hidden dimension on 8 feature maps
        # x = self.mh_attention(input)
        # x = self.norm(x[0]+input)
        # x = self.dec(x)
        # x = self.norm2(input)
        x = self.linear(input)
        x = self.norm3(x)
        #do some reshaping
        # patch size to extract feature maps at the right position
        x = x[:, :, :].unfold(2,self.patch_size**2,self.patch_size**2)
        x = torch.permute(x, dims=(0,2,1,3))

        #todo: patch size per image size
        x = x.unfold(2,h,h).unfold(3,self.patch_size,self.patch_size)#n patches, patch size

        #8 feauture maps
        x = x.reshape((x.shape[0],8, x.shape[2]*x.shape[3], x.shape[4]*x.shape[5]))
        #concate 10->6
        #need 6 feature maps
        return x

class ViT(nn.Module):
    def __init__(self, cfg):
        super(ViT, self).__init__()#load a config for sizes
        self.patch_size = cfg.patch_size
        self.encoder = Encoder(cfg.encoder, hidden_d=cfg.hidden_d, patch_size=self.patch_size)#upscaling failed try downscaling; Discarded V4
        self.decoder = Decoder(cfg.decoder, hidden_d=cfg.hidden_d, patch_size=self.patch_size)#downscaling works try further
        #V4 worked best hiddend 400
        #get and initialize defined activation
        self.activation = getattr(activations, cfg.activation)()

    def forward(self, input):
        b,h,w = input.shape
        latent = self.encoder(input)
        image_space = self.decoder(latent, h//self.patch_size)

        out = self.activation(image_space)
        return out