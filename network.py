from torch import nn
import torch
import torch.nn.functional as F
from math import sqrt


class Activation(nn.Module):
    def __init__(self,):
        super().__init__()
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        input[:, 0] = self.sig(input[:, 0])
        input[:, 1:3] = self.tanh(input[:, 1:3])
        input[:, 3:5] = self.sig(input[:, 3:5])*2
        return input
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

def scaled_dot_product_attention(query, key, value):
    query,key,value = [torch.permute(x, (1,2,0,3)) for x in (query,key,value)]
    dim_k = query.size(-1)
    scores = query @ key.transpose(-1,-2) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.permute(weights @ value, (2,0,1,3))

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
        self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        #todo: embed_dim = 8*num_heads = 8
        embed_dim = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
        [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.linear_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class U_Net(nn.Module):
    def __init__(self,inp=3):
        super().__init__()
        size=.5
        self.down1 = Down(inp,64*size,2)#30
        self.down2 = Down(64*size, 128*size, 2)  # 15
        self.down3 = Down(128*size, 256*size, 3)  # 5
        self.up1 = Up((256+128)*size, 128*size, scale=3)
        self.up2 = Up((128+64)*size, 64*size, )
        self.up3 = Up(64*size+inp, 32*size, )
        self.final = nn.Conv2d(32*size,8,3,padding="same")

    def forward(self, input):
        x1 = self.down1(input)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.up1(x3,x2)
        x5 = self.up2(x4,x1)
        x6 = self.up3(x5,input)
        out = self.final(x6)
        return out


class AttentionUNet(nn.Module):
    def __init__(self,inp=3):
        #todo: use fold instead of 3 input dims -> less data

        super().__init__()
        #works
        size=1
        self.down1 = Down(inp,64*size,2)#30
        self.down2 = Down(64*size, 128*size, 2)  # 15
        self.down3 = Down(128*size, 256*size, 3)  # 5
        conf = {"hidden_size": 512*size, "intermediate_size": 64*size, "num_attention_heads":8}
        self.down4 = Down(256*size, 512*size, 5)  # 5

        self.ff = FeedForward(conf)
        #todo: use feed forward on higher scale on batch
        self.attention = MultiHeadAttention(conf)
        self.up0 = Up((512+256)*size, 256*size, scale=5)

        self.up1 = Up((256+128)*size, 128*size, scale=3)
        self.up2 = Up((128+64)*size, 64*size, )
        self.up3 = Up(64*size+inp, 32*size, )
        self.final = nn.Conv2d(32*size,8,3,padding="same")

    def forward(self, input):
        x1 = self.down1(input)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.ff(x4.squeeze())
        x6 = self.up0(x5.unsqueeze(-1).unsqueeze(-1),x3)
        x7 = self.up1(x6,x2)
        x8 = self.up2(x7,x1)
        x9 = self.up3(x8,input)
        out = self.final(x9)
        return out


class AttentionUNetV2(nn.Module):
    def __init__(self,inp=3):
        super().__init__()
        #works
        size=1
        self.down1 = Down(inp,64*size,2)#30
        self.down2 = Down(64*size, 128*size, 2)  # 15
        self.down3 = Down(128*size, 256*size, 3)  # 5
        conf = {"hidden_size": 256, "intermediate_size": 64, "num_attention_heads":8}

        self.ff = FeedForward(conf)
        self.attention = MultiHeadAttention(conf)
        self.up0 = Up((512+256)*size, 256*size, scale=5)

        self.up1 = Up((256+128)*size, 128*size, scale=3)
        self.up2 = Up((128+64)*size, 64*size, )
        self.up3 = Up(64*size+inp, 32*size, )
        self.final = nn.Conv2d(32*size,8,3,padding="same")

    def forward(self, input):
        x1 = self.down1(input)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = torch.permute(x3, (0,2,3,1))
        x5 = self.ff(x4)

        x5 = self.attention(x5)
        x6 = torch.permute(x5, (0,3,1,2))

        x7 = self.up1(x6,x2)
        x8 = self.up2(x7,x1)
        x9 = self.up3(x8,input)
        out = self.final(x9)
        return out

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = Activation()
        #double UNet >> normal Unet
        self.z_size =9

        self.unet = AttentionUNet(inp=self.z_size)
        self.unet2 = AttentionUNet(8)

    def forward(self, input):
        input = F.pad(input=input, pad=( 0,0,0,0,(self.z_size-1)//2, (self.z_size-1)//2,), mode='constant', value=0)
        x = torch.permute(input.unfold(0,self.z_size,1), (0,3,1,2))
        x6 = self.unet(x)
        x7 = self.unet2(x6)
        out = self.act(x7)
        return out


class Network2(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = Activation()
        #double UNet >> normal Unet
        self.z_size =9

        self.unet = AttentionUNetV2(inp=9)



    def forward(self, input):
        input = F.pad(input=input, pad=( 0,0,0,0,(self.z_size-1)//2, (self.z_size-1)//2,), mode='constant', value=0)
        x = torch.permute(input.unfold(0,self.z_size,1), (0,3,1,2))
        x6 = self.unet(x)
        #7 = self.unet2(x6)
        #x7 = self.unet3(x7)
        out = self.act(x6)
        return out