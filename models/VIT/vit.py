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
        return x#self.dropout(x)


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
    def __init__(self, chw=(100,60,60),patch_size=10, hidden_d=100):
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

        self.grou_norm = nn.GroupNorm(6,36)
        #self.linear_mapper.weight.data = torch.permute(gaussian_initializer(),(1,0))
        #self.linear_mapper.bias.data.fill_(0.0)
        #layer norm before embedding?
        #discard image normalization and rather increase pos embedding

        # 2) Learnable classification token todo test if this works
        #self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        # 3) Positional embedding
        self.pos_embed = PositionalEncoding(self.hidden_d)
        self.group_norm2 = nn.GroupNorm(8,8)



    def forward(self,images):
        images = torch.log(nn.ReLU()(images + images.min()) + 0.1)
        images /= images.max()

        images = self.conv(images[:,None])

        images = self.unet(images)

        #todo: apply residual connection?

        patches = torch.permute(images, (0,2,3,1))
        patches = patchify(patches, self.patch_size)
        tokens = self.linear_mapper(patches)
        # Adding classification token to the tokens
        #tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # Adding positional embedding
        # apply positional encoding over batch size
        out = self.pos_embed(tokens)
        out = self.grou_norm(out)
        return out,self.group_norm2(images)


class Encoder(nn.Module):
    def __init__(self, cfg, hidden_d=128, mlp_ratio=2, patch_size=10):
        #todo: restructure
        super(Encoder, self).__init__()
        #min U-Net
        #embedding + MHA
        #decoding
        #todo: this could be sequential
        #todo: define in VIT
        self.embedding = ImageEmbedding(hidden_d=hidden_d, patch_size=patch_size)
        self.mha = MHA(embed_dim=hidden_d)#is normed
        #self.norm2 = nn.LayerNorm(hidden_d)#todo: deactivates norms in encoder
        self.mlp = MLP2(embed_dim=hidden_d, mlp_ratio=mlp_ratio)#is normed
        #self.mh_attention2 = MHA(embed_dim=hidden_d)
        #self.norm3 = nn.LayerNorm(hidden_d)

    def forward(self, input):
        x1,images = self.embedding(input)
        #x = self.mha(x1)
        #x = self.mlp(x)
        #residual connection
        return x1,images
class CADecoder(nn.Module):
    def __init__(self, cfg, hidden_d=128, feature_map_d=8, mlp_ratio=2, patch_size=10):
        super().__init__()

        self.patch_size = patch_size
        #todo: bottleneck becuse hidden_d to small?
        self.groupnorm = nn.GroupNorm(4, 8)
        self.linear = nn.Linear(hidden_d, patch_size**2*feature_map_d)
        self.norm3 = nn.LayerNorm(patch_size**2 * feature_map_d)
        self.cross_attention = CrossAttentionBlock()
        self.activation = activations.GMMActivation()
        #todo: this is in test
        self.final = nn.Conv2d(8,8,1,padding="same")

    def forward(self, x, h):
        #fig,axs = plt.subplots(2)
#        axs[0].imshow(x[0].cpu().detach().numpy())
        x = self.linear(x) #400-> 800 might cause bottleneck
        # axs[1].imshow(x[0].cpu().detach().numpy())
        # plt.show()

        x1 = self.norm3(x)
        #x = x.unfold(2, self.patch_size ** 2, self.patch_size ** 2)
        #n,c1,c2,w = x.shape
        #x = x.reshape((n,c1*c2,self.patch_size,self.patch_size))
        #todo: find out why cad decoder leads to artefacts
        x = decoder_unpatch(x1, self.patch_size)
        #x1 = decoder_unpatch(x1, self.patch_size)
        x = self.cross_attention(x, self.activation(x))
        x = decoder_patchify(x, 36)
        x = unpatchify(x+x1, self.patch_size, n_channels=8)
        x = self.groupnorm(x)
        return self.final(x)

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
        self.mha = MHA(embed_dim=8,head_dim=4)
        self.mlp = MLP2(embed_dim=8, mlp_ratio=12)#added in V1

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
        self.final = nn.Conv2d(8,8,1,padding="same")
    def forward(self, input, images):
        #todo: map larger hidden dimension on 8 feature maps
        #x = self.linear(input)
        #x = self.norm3(x)
        #do some reshaping
        # patch size to extract feature maps at the right position
        b,c,h,w = images.shape

        x = torch.permute(images,(0,2,3,1))
        x = x.reshape(b,h*w,c)
        x = self.mha(x)
        x = self.mlp(x)
        x = x.reshape(b,h,w,c)
        x = torch.permute(x,(0,3,1,2))

        #x = unpatchify(x, self.patch_size, n_channels=8)
        #concate 10->6
        #need 6 feature maps
        return self.final(x+images)

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
        latent,images = self.encoder(input)
        image_space = self.decoder(latent,images)

        out = self.activation(image_space)
        return out

class ViT2(nn.Module):
    def __init__(self, cfg):
        super(ViT2, self).__init__()#load a config for sizes
        self.patch_size = cfg.patch_size
        self.encoder = Encoder(cfg.encoder, hidden_d=cfg.hidden_d, patch_size=self.patch_size)#upscaling failed try downscaling; Discarded V4
        self.cadecoder = CADecoder(cfg.decoder, hidden_d=cfg.hidden_d, patch_size=self.patch_size)#downscaling works try further
        #V4 worked best hiddend 400
        #get and initialize defined activation
        self.activation = getattr(activations, cfg.activation)()

    def forward(self, input):
        b,h,w = input.shape
        latent = self.encoder(input)
        image_space = self.cadecoder(latent, h//self.patch_size)

        out = self.activation(image_space)
        return out