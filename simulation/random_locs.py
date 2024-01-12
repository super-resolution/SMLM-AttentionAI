import os
import numpy as np

def create_locs( im_size: np.ndarray):
    """
    create random points to train a neural network
    :param average_points: average points per frame
            frames: number of rendered frames
    :return:
    """
    locs = np.random.random_sample((300000,2))*im_size
    return locs

if __name__ == '__main__':
    position_data = create_locs(np.array([60,60]))
    path = "../data/random_highpower2"
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(path+"/coords.npy", position_data)
