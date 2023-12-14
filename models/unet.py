from models.layers import *


class UNet(nn.Module):
    """
    Implementing a Modular U-Net with flexible layer depth and sizes will be enhaced to take a config parameter
    """
    def __init__(self):
        super().__init__()
        size = 1
        # 1. Define layer sizes in down/up path
        d_size = (8,16,32)
        # 2. Get amount of layers
        n_layers = len(d_size)
        # 3. Create DoubleConv+Maxpool from d_size[i]->d_size[i] feature maps thats down path halving image size n_layer times
        self.d_path = nn.ModuleList([Down(d_size[i], d_size[i+1], 2) for i in range(n_layers-1)])
        # 4. Create UpPath concatenating the facing feature maps of the down path with residual conenctions
        self.u_path = nn.ModuleList([Up(d_size[i] + d_size[i-1], d_size[i-1]) for i in range(n_layers-1,0,-1)])
        # 5. finish with a 2d convolution
        self.final = nn.Conv2d(8 * size, 8, 3, padding="same")

    def forward(self, x):
        "Forward path of unet"
        stack = []
        for down in self.d_path:
            stack.append(x)
            x = down(x)
        for up in self.u_path:
            x = up(x, stack.pop())
        return self.final(x)
