import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu, edges
from skimage import feature
from skimage.transform import rescale
from skimage.morphology import dilation
import scipy


class SauerLab():
    @staticmethod
    def create_binary_image(im):
        new = np.sum(im, axis=-1)
        thresh = threshold_otsu(new)
        bin = new < thresh
        bin = dilation(bin ^ feature.canny(new, 2))

        return bin

    def generate_sauerlab_pointcloud_all(self):
        """
        requires 32x59 grid
        :return:
        """
        px_size = 100

        # large logo 5900x3200 px
        im_logo = Image.open(r"resources/Logo_weiß.png")
        im_logo = np.array(im_logo)

        bin_logo = self.create_binary_image(im_logo)
        # get indices from binary image
        indices = np.array(np.where(bin_logo == 1))
        # image is big enough to only use 10% of points
        subset = indices.T[np.where(np.random.binomial(1, 0.1, indices[0].shape[0]))]
        loc = subset.astype(np.float32) / px_size
        return np.array(loc)


if __name__ == '__main__':
    s = SauerLab()
    position_data = s.generate_sauerlab_pointcloud_all()
    np.save("data/lab_logo.npy", position_data)
