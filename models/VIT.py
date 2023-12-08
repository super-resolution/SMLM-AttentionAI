import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor
import math
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import numpy as np
class Activation(nn.Module):
    def __init__(self,):
        super().__init__()
        #todo: determine multiplier in config
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        input[:, 0] = self.sig(input[:, 0])
        input[:, 1:3] = self.tanh(input[:, 1:3])
        input[:, 3:5] = self.sig(input[:, 3:5])*2
        input[:, 6] = self.sig(input[:, 6])*2
        input[:, 7] = self.sig(input[:, 7])*3

        return input
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 36):
        super().__init__()#todo: adapt this to VIT need 3 positional encodings
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)*100
        pe[0, :, 1::2] = torch.cos(position * div_term)*100#todo: should be x and y?
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1)]
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
class ImageEmbedding(torch.nn.Module):
    def __init__(self, chw=(100,60,60),patch_size=(10,10), hidden_d=128):
        super(ImageEmbedding, self).__init__()
        # Attributes
        double_conv_width = 8
        self.chw = chw # (C, H, W)
        #assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        #assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = patch_size
        self.n_patches = (chw[1]//self.patch_size[0],chw[2]//self.patch_size[1])
        self.hidden_d = hidden_d
        # 1) Linear mapper
        self.input_d = int(self.patch_size[0] * self.patch_size[1]*double_conv_width)
        #todo: project on 8 channels with double conv
        self.conv = DoubleConv(1,double_conv_width)
        #todo: add some convolutional layers here?
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        #todo: initial values for linea mapping?



        #self.linear_mapper.weight.data = torch.permute(gaussian_initializer(),(1,0))
        #todo: is this working? probably not so discard?
        #self.linear_mapper.bias.data.fill_(0.0)
        #layer norm before embedding?
        #discard image normalization and rather increase pos embedding

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        # 3) Positional embedding
        self.pos_embed = PositionalEncoding(self.hidden_d)

    def patchify(self, images):
        patches = images.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1])
        patches = torch.reshape(patches, (patches.shape[0], patches.shape[1] * patches.shape[2], -1))
        return patches

    def forward(self,images):
        images = self.conv(images[:,None])
        images = torch.permute(images, (0,2,3,1))
        patches = self.patchify(images)
        #todo: apply skip connection?
        tokens = self.linear_mapper(patches)
        # Adding classification token to the tokens
        #tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # Adding positional embedding
        out = self.pos_embed(tokens) #todo: apply positional encoding over batch size
        return out


class MHA(nn.Module):
    def __init__(self, embed_dim=128, head_dim=8):
        super(MHA, self).__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=head_dim,dropout=0.1, batch_first =True)
    def forward(self, input):
        #Layer norm
        x = self.norm(input)
        #Compute Query Key and Value
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        #residual connection + multihead attention
        return self.mha(q,k,v,need_weights=False)

class Encoder(nn.Module):
    def __init__(self, hidden_d=128, mlp_ratio=2):
        super(Encoder, self).__init__()
        self.embedding = ImageEmbedding(hidden_d=hidden_d)#todo: needs image dimensions
        self.mh_attention = MHA(embed_dim=hidden_d)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )
        self.mh_attention2 = MHA(embed_dim=hidden_d)
        self.norm3 = nn.LayerNorm(hidden_d)
    def forward(self, input):
        x = self.embedding(input)
        x2 = self.mh_attention(x)
        x2 = x+x2[0]
        x = self.norm2(x2)
        x = self.mlp(x)
        x = x +x2
        x = self.norm3(x)
        #residual connection
        return x+x2

class Decoder(nn.Module):
    def __init__(self, hidden_d=128,feature_map_d=8, mlp_ratio=2, patch_size=(10,10)):
        super(Decoder, self).__init__()
        #todo: input dimension by config/patch size
        self.mh_attention = MHA(embed_dim=hidden_d)
        self.norm = nn.LayerNorm(hidden_d)
        self.dec = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d,hidden_d))
        self.norm2 = nn.LayerNorm(hidden_d)


        self.linear = nn.Linear(hidden_d, patch_size[0]*patch_size[1]*feature_map_d)
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

        self.norm3 = nn.LayerNorm(patch_size[0]*patch_size[1]*feature_map_d)

    def forward(self, input):
        #todo: map larger hidden dimension on 8 feature maps
        # x = self.mh_attention(input)
        # x = self.norm(x[0]+input)
        # x = self.dec(x)
        # x = self.norm2(input)
        x = self.linear(input)
        x = self.norm3(x)
        #do some reshaping
        # patch size to extract feature maps at the right position
        x = x[:, :36, :].unfold(2,100,100)
        x = torch.permute(x, dims=(0,2,1,3))
        x = x.unfold(2,6,6).unfold(3,10,10)#n patches, patch size
        #8 feauture maps
        x = torch.reshape(x,(-1,8, x.shape[2]*x.shape[3], x.shape[4]*x.shape[5]))

        #concate 10->6
        #need 6 feature maps
        return x

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()#todo: load a config for sizes
        self.encoder = Encoder(hidden_d=1600)
        self.decoder = Decoder(hidden_d=1600)
        self.activation = Activation()

    def forward(self, input):
        latent = self.encoder(input)
        image_space = self.decoder(latent)
        out = self.activation(image_space)
        return out