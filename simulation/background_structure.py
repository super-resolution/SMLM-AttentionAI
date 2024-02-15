import cv2
import numpy as np
from numpy.random import default_rng


def create_images(n_images:int=10000, seed_val:int=73, height:int=60, width:int=60)->np.ndarray:
    """
    Create smooth random background images

    Example usage:
    >>> from tifffile.tifffile import imwrite
    >>> path = "../data/lab_logo2"
    >>> images = create_images()
    >>> imwrite(path + "/bg_images.tif", images)
    :param n_images: Number of images to create
    :param seed_val: Seed value for random number generator
    :return: batch of random background images
    """
    rng = default_rng(seed_val)
    #image width and height
    #create empty array to fill with images
    images = np.zeros((n_images,height,width))
    for i in range(n_images):
        #create random noise image
        noise = rng.integers(0, 255, (height,width), np.uint8, True)
        #blur the noise image to control the size
        blur = cv2.GaussianBlur(noise, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT).astype(np.float32)
        #Set min to zero
        m = blur.max()
        blur -= blur.min()
        blur /= blur.max()
        blur *= m
        #put image in array
        images[i] = blur
    return images




