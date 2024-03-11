import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import get_original_cwd
from tifffile.tifffile import imwrite, imread

from markov_chain import SimpleMarkovModel
from noise_simulations import Simulation

from simulation.background_structure import create_images
from simulation.random_locs import create_locs
from simulation.data_augmentation import DropoutBox
from simulation.simulator import Simulator



@hydra.main(config_name="base.yaml", config_path="../cfg/")
def my_app(cfg):
    # A batch should contain 250 images
    # Should end up in ~100 000 frames
    # Variate density and number of localisations
    # Define off time per batch
    # Keep microscope static
    #todo: restrucure
    #todo: cls from photon traces
    path = get_original_cwd() + "\\" + cfg.dataset.path + "\\" + cfg.dataset.name
    # check if bg images and coords are already defined in the target folder
    if os.path.exists(path):
        bg_images = imread(path + "/bg_images.tif")
        o_arr = np.load(path + f"/coords.npy", allow_pickle=True)
    else:
        os.mkdir(path)
        # Limited amount is fine since these undergo a distribution
        bg_images = create_images(cfg.dataset.n_batches*cfg.dataset.batch_size//4,seed_val=cfg.dataset.seed)
        imwrite(path + "/bg_images.tif", bg_images)
        #c = SauerLab()
        o_arr = create_locs(cfg.dataset.n_pix)
        np.save(path + "/coords.npy", o_arr)

    s = Simulator(cfg.dataset.batch_size, cfg.dataset.n_pix, path, cfg.microscope, cfg.emitter.emitter_density,
                  cfg.emitter.off_time, o_arr, cfg.dataset.mode)
    s(bg_images, cfg.dataset.n_batches)

    # #initialize standard parameters
    # # Emitters have 50% chance to switch into an off state and off_state_prob chance to stay there
    # # compute switching prob with density if there is a denstiy
    # n_loc = o_arr.shape[0]
    # # compute area of image
    # area = (cfg.dataset.n_pix*cfg.microscope.px_size/1000)**2
    #
    # #either estimate off state_prob for density
    # off_state_prob = np.around(cfg.emitter.emitter_density/(n_loc/area),9)
    #
    # #or adjust_n_loc instead of off_state prob
    # #todo: only use this if off_time is given
    # coeff = off_state_prob*cfg.emitter.off_time
    #
    # off_state_prob = 1/cfg.emitter.off_time
    #
    #
    # #transition_distribution = cfg.emitter.switching_probability
    # transition_distribution = [[0.5,0.5],[1-off_state_prob,off_state_prob]]
    #
    # # final obeservation as joint distribution
    # observation_distribution = [[[0.0, 1.0], [1.0, 0.0]]]
    #
    # # We can combine these distributions into a hidden Markov model with n = 30 frames:
    # frames = []
    # ground_truth = []
    # idx = 0
    # offsets = []
    # #initialize simulation for noise dictionary
    # sim = Simulation.from_dict(cfg.microscope,device=cfg.device).to(cfg.device)
    # # load bg images to gpu
    # bg_t = torch.tensor(bg_images, device="cuda", dtype=torch.int16)
    #
    # dropout_box = DropoutBox(o_arr.max())
    # for i in range(cfg.dataset.n_batches):
    #     #adjust number of points if they dont fit off_time and density
    #     if coeff <1:
    #         cur_arr = o_arr[np.where(np.random.binomial(1, coeff, o_arr.shape[0]))]
    #     #todo: works only if coeff is adjusted
    #     initial_distribution = [[0, 1]]*cur_arr.shape[0]
    #
    #     #Use SimpleMarkovModel to approximate the switching behavior
    #     chain = SimpleMarkovModel(initial_distribution, transition_distribution, observation_distribution, n_frames=cfg.dataset.batch_size, device=cfg.device).to(cfg.device)
    #     ch = chain()
    #     #load points to gpu
    #     arr = torch.tensor(cur_arr, device="cuda", dtype=torch.float32)
    #
    #
    #     for i,indices in enumerate(ch):
    #         #pick random bg_image
    #         #select one random background image
    #         bg_index = torch.randint(0,len(bg_t),(1,),device="cuda")
    #         #update dropout box every 50 frames
    #         if i%50==0:
    #             dropout_box.update_box()
    #         #load remaining coords into simulation
    #         #todo: make everything torch
    #         arr_dropout = dropout_box.forward(arr[indices])
    #         #random intensity not used right now should come with better simulations
    #         I = torch.rand(arr_dropout.shape[0], device="cuda")/2+.5
    #         #concatenate data
    #         dat = torch.concat([arr_dropout,I[:,None],bg_index.repeat(arr_dropout.shape[0],1)], dim=1)
    #         #feed bg image here and data
    #         frame = sim(dat, bg_t[bg_index[0]])
    #         frames.append(frame.cpu().squeeze().numpy())
    #         #save ground truth
    #         data = dat.cpu().numpy()
    #         ground_truth.append(data)#todo: add bg_t here
    #         offsets.append(idx)
    #         idx += data.shape[0]
    # offsets.append(idx)
    # np.save(path + f"/ground_truth", np.concatenate(ground_truth,axis=0))
    # np.save(path + f"/offsets", offsets)
    #
    # imwrite(path + "/images.tif", frames)
    # #show some frames
    # for i,frame in enumerate(frames):
    #     if i+900%1000 == 0:
    #         plt.scatter(ground_truth[i][:,0],ground_truth[i][:,1])
    #         plt.imshow(frame.cpu().squeeze())
    #         plt.show()


if __name__ == '__main__':
    my_app()

