import hydra
import matplotlib.pyplot as plt
import torch
from tifffile.tifffile import imread
import numpy as np
from utility.emitters import Emitter
from models.network import FFTAttentionUNet
from utility.visualization import plot_emitter_set


@hydra.main(config_name="eval.yaml", config_path="cfg")
def myapp(cfg):
    device = cfg.network.device
    dataset_name = cfg.dataset.name
    dataset_offset = cfg.dataset.offset
    dtype = getattr(torch, cfg.network.dtype)


    images = imread(r"D:\Daten\Patrick\STORMHD\647\COS7_Phalloidin_ATTO647_1_200_2perHQ_1.tif")[0:10000,0:60,0:60].astype(np.int32)
    images -= images.min()

    #reshape for temporal context
    images = torch.tensor(images, dtype=dtype, device=device)/10

    model_path = 'trainings/model_FFTAttentionV2'
    print(model_path)

    net = FFTAttentionUNet()
    #opt_cls = getattr(torch.optim, cfg.optimizer.name)
    #opt = opt_cls(net.parameters(), **cfg.optimizer.params)

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    #opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(loss)


    net.eval()
    with torch.no_grad():
        out_data = net(images).cpu()

    out_data = out_data.numpy()
    plt.imshow(out_data[0,0])
    plt.show()
    dat = Emitter.from_result_tensor(out_data[:, ], 0.9)
    plot_emitter_set(dat)

if __name__ == '__main__':
    myapp()