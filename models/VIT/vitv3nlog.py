
from models import activations
from models.layers import *
from models.unet import UNet

class ImageEmbedding(torch.nn.Module):
    def __init__(self, chw=(100,60,60),patch_size=10, hidden_d=100):
        super(ImageEmbedding, self).__init__()
        # Attributes
        double_conv_width = 8
        # 1) Linear mapper
        # project on 8 channels with double conv
        self.conv = DoubleConv(1,double_conv_width)
        #takes 8 feauture maps
        self.unet = UNet()
        #todo: positional encode axis 0
        self.group_norm2 = nn.GroupNorm(8,8)



    def forward(self,images):
        images /= images.max()

        images = self.conv(images[:,None])

        images = self.unet(images)

        return self.group_norm2(images)


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
        #self.norm2 = nn.LayerNorm(hidden_d)#todo: deactivates norms in encoder
        #self.mh_attention2 = MHA(embed_dim=hidden_d)
        #self.norm3 = nn.LayerNorm(hidden_d)

    def forward(self, input):
        images = self.embedding(input)
        #x = self.mha(x1)
        #x = self.mlp(x)
        #residual connection
        return images

class Decoder(nn.Module):
    def __init__(self, cfg, hidden_d=128, feature_map_d=8, mlp_ratio=2, patch_size=10):
        super(Decoder, self).__init__()

        self.final = nn.Conv2d(8,8,1,padding="same")

    def forward(self, images):
        return self.final(images)

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
        images = self.encoder(input)
        image_space = self.decoder(images)

        out = self.activation(image_space)
        return out