import cv2
import matplotlib.pyplot as plt
import numpy as np
from tifffile.tifffile import TiffWriter


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
    with TiffWriter('../temp.tif', bigtiff=True) as tif:
        tif.save(downsampled)
    cm = plt.get_cmap('hot')
    v = cm(downsampled / 255)
    v[:, :, 3] = 255
    v[-25:-20, 10:110, 0:3] = 1

    # array = np.log(array+1)
    plt.imshow(array, cmap='hot')
    plt.show()