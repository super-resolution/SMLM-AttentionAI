import cv2
import matplotlib.pyplot as plt
import numpy as np
from tifffile.tifffile import TiffWriter
import torch
import torch.distributions as td
from PIL import Image

def plot_emitter_gmm(emitters):
    #todo: render on div
    mesh = torch.meshgrid([torch.arange(0, 600) + .5, torch.arange(0, 600) + .5])
    grid = torch.stack(mesh, 0)
    cat = td.Categorical(torch.tensor(emitters.p[0:5000]))
    comp = td.Independent(td.Normal(torch.tensor(emitters.xyz[:5000]/10), torch.tensor(emitters.sigxsigy[:5000]*10)), 1)
    gmm = td.mixture_same_family.MixtureSameFamily(cat, comp)
    v = torch.exp(gmm.log_prob(torch.transpose(grid,0,2)))
    x = 0

    plt.imshow(v,cmap="hot")
    plt.show()
    #sample grid


def plot_emitter_set(emitters,save_name="temp", frc=False):
    """
    Image from emitter set class
    :param emitters:
    :return:
    """
    # todo: show first and second half for FRC
    # data_in = data_in[np.where(data_in[:,2]<data_in[:,2].max()/3)]
    # data_in = data_in[1::2]

    localizations = emitters.xyz  # +np.random.random((data_in.shape[0],2))
    #array = np.zeros((4000,6000))
    array = np.zeros((int(localizations[:, 0].max()) + 1, int(localizations[:, 1].max()) + 1))  # create better rendering...
    #sort by sigma
    #sum up images
    #todo crop stuff
    for i in range(localizations.shape[0]):
        if int(localizations[i, 0])<array.shape[0] and int(localizations[i, 1]) < array.shape[1]:
            array[int(localizations[i, 0]), int(localizations[i, 1])] += 300# * emitters.photons[i]

    array = cv2.GaussianBlur(array, (21, 21), 0)
    # array -= 10
    array = np.clip(array, 0, 255)
    downsampled = cv2.resize(array, (int(array.shape[1] / 10), int(array.shape[0] / 10)), interpolation=cv2.INTER_AREA)

    # todo: make 10 px scalebar
    # with TiffWriter(f'{save_name}.tif', bigtiff=True) as tif:
    #     tif.save(downsampled)
    cm = plt.get_cmap('hot')
    v = cm(downsampled / (downsampled.max()/4))
    v[:, :, 3] = 255
    v[-25:-20, 10:110] = 1

    im = Image.fromarray((v[:,:,0:3]*255).astype(np.uint8),"RGB")
    im.save(f'{save_name}.jpg')

    # #array = np.log(array+1)
    # plt.imshow(downsampled, cmap='hot')
    # plt.show()