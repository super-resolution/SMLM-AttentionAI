import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra

from markov_chain import HiddenMarkovModel
from noise_simulations import Simulation
from tifffile.tifffile import imwrite
td = torch.distributions


@hydra.main(config_name="base.yaml", config_path="cfg/")
def my_app(cfg):
    path = "data/random"
    arr = np.load(path + "/coords.npy", allow_pickle=True)
    #initialize standard parameters
    initial_distribution = [[1, 0]]*arr.shape[0]

    # Emitters have 50% chance to switch into an off state and 0.999 chance to stay there

    transition_distribution = cfg.emitter.switching_probability

    # final obeservation as joint distribution
    observation_distribution = [[[0.0, 1.0], [1.0, 0.0]]]

    # We can combine these distributions into a hidden Markov model with n= 30 frames:


    chain = HiddenMarkovModel(initial_distribution, transition_distribution, observation_distribution, n_frames=10000, device=cfg.device).to(cfg.device)
    sim = Simulation(device=cfg.device).to(cfg.device)

    ch = chain()
    ch = np.array(ch, dtype=np.int8)
    arr = torch.tensor(arr, device="cuda", dtype=torch.float32)
    frames = []
    for observations in ch:
        indices = np.where(observations[:,0]==1)
        frame = sim(arr[indices])
        frames.append(frame.cpu().squeeze())
    np.save(path + "/indices", ch)
    imwrite(path + "/images.tif", frames)
    plt.imshow(frame.cpu().squeeze())
    plt.show()

if __name__ == '__main__':
    my_app()

