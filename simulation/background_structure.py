import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from tifffile.tifffile import imwrite


# read input image
n_images = 10000
# define random seed to change the pattern
seedval = 78
rng = default_rng(seedval)
h,w = 60,60
images = np.zeros((n_images,h,w))
for i in range(n_images):
    img = np.zeros((h,w))



    # create random noise image
    noise = rng.integers(0, 255, (h,w), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT).astype(np.float32)
    m = blur.max()
    blur -= blur.min()
    blur /= blur.max()
    blur *= m

    images[i] = blur
path = "../data/random_highpower_test"
imwrite(path + "/bg_images.tif", images)


