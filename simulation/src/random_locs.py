import os
import numpy as np

def create_locs(im_size: np.ndarray, n: int = 30000000) -> np.ndarray:
    """
    Generate random locations within an image size for training a neural network.

    Parameters:
        im_size (np.ndarray): Array representing the size of the image.
        n (int): Number of random points to generate. Defaults to 30,000,000.

    Returns:
        np.ndarray: Array of shape (n, 2) containing random points within the image size.
    """
    locs = np.random.random_sample((n, 2)) * im_size
    return locs


if __name__ == '__main__':
    image_size = np.array([60,60])
    position_data = create_locs(image_size)
    path = "../data/random_highpower2"
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(path+"/coords.npy", position_data)
