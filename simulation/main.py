import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra

from hydra.utils import get_original_cwd, to_absolute_path
from markov_chain import HiddenMarkovModel
from noise_simulations import Simulation
from tifffile.tifffile import imwrite,imread
td = torch.distributions


def create_box():
    box = np.random.rand(2)
    x_min = box.min()
    x_max = box.max()
    box = np.random.rand(2)
    y_min = box.min()
    y_max = box.max()
    return {"x_min":x_min,"x_max":x_max,"y_min":y_min,"y_max":y_max}

@hydra.main(config_name="base.yaml", config_path="../cfg/")
def my_app(cfg):
    # todo: create batches of datasets under different conditions
    # a batch should contain 250 images
    # should end up in ~100 000 frames
    #variate density and number of localisations
    #use lower number of localisations

    #keep microscope static
    path = get_original_cwd() + "\\" + cfg.dataset.path + "\\" + cfg.dataset.name
    #limited amount is fine since these undergo a distribution
    bg_images = imread(path + "/bg_images.tif")
    o_arr = np.load(path + f"/coords.npy", allow_pickle=True)
    #initialize standard parameters
    initial_distribution = [[0, 1]]*o_arr.shape[0]
    #load image and add as background

    # Emitters have 50% chance to switch into an off state and off_state_prob chance to stay there
    # compute switching prob with density if there is a denstiy
    n_loc = o_arr.shape[0]
    area = (cfg.dataset.n_pix/10)**2
    off_state_prob = np.around(cfg.emitter.emitter_density/(n_loc/area),5)

    #transition_distribution = cfg.emitter.switching_probability
    transition_distribution = [[0.5,0.5],[1-off_state_prob,off_state_prob]]

    # final obeservation as joint distribution
    observation_distribution = [[[0.0, 1.0], [1.0, 0.0]]]

    # We can combine these distributions into a hidden Markov model with n = 30 frames:

    #todo: gather localisations in the on state with binning of intensity
    chain = HiddenMarkovModel(initial_distribution, transition_distribution, observation_distribution, n_frames=1000, device=cfg.device).to(cfg.device)
    sim = Simulation(device=cfg.device).to(cfg.device)

    ch = chain()

    #load images to gpu
    arr = torch.tensor(o_arr, device="cuda", dtype=torch.float32)
    bg_t = torch.tensor(bg_images, device="cuda", dtype=torch.int16)

    frames = []
    ground_truth = []
    idx = 0
    offsets = []
    m = o_arr.max()
    for i,indices in enumerate(ch):
        #pick random bg_image
        bg_index = torch.randint(0,len(bg_t),(1,),device="cuda")
        #create data dropout
        if i%300==0:
            box = create_box()
            for k,v in box.items():
                box[k] = v*m
        wheres = [o_arr[indices,0]<box["x_min"],
                  o_arr[indices,0]>box["x_max"],
                  o_arr[indices,1]>box["y_max"],
                  o_arr[indices,1]<box["y_min"]]
        prev = wheres[0]
        for i in range(1,len(wheres)):
            prev = np.logical_or(prev,wheres[i])
        #data augmentation discard data within box
        v = np.where(prev)
        indices = indices[v]
        #load remaining coords into simulation
        I = torch.rand(indices.shape[0], device="cuda")/2+.5
        dat = torch.concat([arr[indices],I[:,None],bg_index.repeat(indices.shape[0],1)], dim=1)
        frame = sim(dat, bg_t[bg_index[0]])#todo: feed bg image here
        frames.append(frame.cpu().squeeze().numpy())
        data = dat.cpu().numpy()
        ground_truth.append(data)#todo: add bg_t here
        offsets.append(idx)
        idx += data.shape[0]
    offsets.append(idx)
    np.save(path + f"/ground_truth", np.concatenate(ground_truth,axis=0))
    np.save(path + f"/offsets", offsets)


    imwrite(path + "/images.tif", frames)
    for i,frame in enumerate(frames):
        if i+900%1000 == 0:
            plt.scatter(ground_truth[i][:,0],ground_truth[i][:,1])
            plt.imshow(frame.cpu().squeeze())
            plt.show()

if __name__ == '__main__':
    my_app()

