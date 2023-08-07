import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile.tifffile import TiffWriter
from tifffile.tifffile import imread

from emitters import Emitter
from network import Network2


def reshape_data(images):
    #add temporal context to additional dimnesion
    dataset = np.zeros((images.shape[0],3,images.shape[1],images.shape[2]))
    dataset[1:,0,:,:] = images[:-1]
    dataset[:,1,:,:] = images
    dataset[:-1,2,:,:] = images[1:]
    return dataset


def plot_emitter_set(emitters, frc=False):
    """
    Image from emitter set class
    :param emitters:
    :return:
    """
    # todo: show first and second half for FRC
    # data_in = data_in[np.where(data_in[:,2]<data_in[:,2].max()/3)]
    # data_in = data_in[1::2]

    localizations = emitters.xyz  # +np.random.random((data_in.shape[0],2))
    array = np.zeros(
        (int(localizations[:, 0].max()) + 1, int(localizations[:, 1].max()) + 1))  # create better rendering...
    for i in range(localizations.shape[0]):
            array[int(localizations[i, 0]), int(localizations[i, 1])] += 300# * emitters.photons[i]


    array = cv2.GaussianBlur(array, (21, 21), 0)
    # array -= 10
    array = np.clip(array, 0, 255)
    downsampled = cv2.resize(array, (int(array.shape[1] / 10), int(array.shape[0] / 10)), interpolation=cv2.INTER_AREA)
    # todo: make 10 px scalebar
    with TiffWriter('temp.tif', bigtiff=True) as tif:
        tif.save(downsampled)
    #
    cm = plt.get_cmap('hot')
    v = cm(downsampled / 255)
    v[:, :, 3] = 255
    v[-25:-20, 10:110, 0:3] = 1

    # array = np.log(array+1)
    plt.imshow(array, cmap='hot')
    plt.show()


@hydra.main(config_name="eval.yaml", config_path="cfg")
def myapp(cfg):
    device = cfg.network.device
    dataset_name = cfg.dataset.name
    dataset_offset = cfg.dataset.offset


    dtype = getattr(torch, cfg.network.dtype)
    arr = np.load("data/"+ dataset_name + "/coords.npy" , allow_pickle=True)


    images = imread("data/"+ dataset_name + "/images.tif")[dataset_offset:]

    #reshape for temporal context
    images = torch.tensor(images, dtype=dtype, device=device)

    model_path = 'trainings/model_AttentionUNet'
    print(model_path)

    net = Network2()
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
    data = Emitter.from_result_tensor(out_data[:,], .2)
    plot_emitter_set(data)

if __name__ == '__main__':
    myapp()
