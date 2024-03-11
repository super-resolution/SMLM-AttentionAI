import torch
import numpy as np
import pickle
from tifffile.tifffile import imwrite, imread
import os
from tqdm import tqdm
from markov_chain import SimpleMarkovModel
from noise_simulations import Simulation

from simulation.background_structure import create_images
from simulation.random_locs import create_locs
from simulation.data_augmentation import DropoutBox


class Simulator():
    def __init__(self, batch_size:int, n_pix:int, path:str, microscope:str,
                 emitter_density:float, off_time:int,
                 positions:np.ndarray, mode:str, device="cuda"):
        """
        Initialize Simulator with configurations for simple markov chain
        :param batch_size: Size of one batch of dependent images
        :param n_pix: Number of pixels in x,y to simulate
        :param path: Path for output data to save to
        :param microscope: Microscope.yaml to save for noise simulations
        :param emitter_density: Emitter density per micro meter²
        :param off_time: Average off time for emitters
        :param positions: Position pool to draw emitters from
        :param device: Computation device
        """
        #todo: add unittests
        self.mode = mode
        self.path = path
        # initialize standard parameters
        # Emitters have 50% chance to switch into an off state and off_state_prob chance to stay there
        # compute switching prob with density if there is a denstiy
        # compute area of image
        self.device = device
        area = (n_pix * microscope.px_size / 1000) ** 2
        n_loc = positions.shape[0]

        # either estimate off state_prob for density
        off_state_prob = np.around(emitter_density / (n_loc / area), 9)

        # or adjust_n_loc instead of off_state prob
        self.coeff = off_state_prob * off_time

        self.off_state_prob = 1 / off_time
        # initialize simulation for noise dictionary
        self.sim =  Simulation.from_dict(microscope, device=self.device).to(self.device)
        #need arr max size here
        self.dropout_box = DropoutBox(positions.max())
        self.batch_size = batch_size
        self.o_arr = positions

    def collect_data(self, path):
        raise NotImplementedError()

    def load_complex_trace(self, n_points:int, n_frames:int, path:str, ):
        #todo: classmethod with trace path
        with open(path, 'rb') as f:
            traces = pickle.load(f)
        #per point select random trace and random start with window n_frames
        #todo: 80000 is number of points in trace
        trace_idx = np.random.choice(80000, n_points, False)
        #todo: 1300 is number of simulated frames
        trace_idstart = np.random.randint(100, 1300-n_frames)
        out = []
        for i in tqdm(range(n_frames)):
            values = np.array(list(traces[trace_idstart+i].items()))
            if np.any(values):
                ind = np.where(np.in1d(trace_idx, values[:,0],assume_unique=True))
                v_ind = np.where(np.in1d(values[:,0],trace_idx,assume_unique=True))
                #todo take the right index value in range(n_points)
                v = values[v_ind]
                v[:,0] = ind[0]
                out.append(v)
        return out
        #todo: collect frames in trace_ids range

    def simulate_simple_trace(self, off_state_prob:float, n_points:int, n_frames:int):
        # transition_distribution = cfg.emitter.switching_probability

        # Use SimpleMarkovModel to approximate the switching behavior
        chain = SimpleMarkovModel(off_state_prob,n_points,
                                  n_frames=n_frames, device=self.device).to(self.device)
        return chain()

    def __call__(self, bg_images:np.ndarray, n_batches:int):


        # We can combine these distributions into a hidden Markov model with n = 30 frames:
        frames = []
        ground_truth = []
        idx = 0
        offsets = []

        # load bg images to gpu
        bg_t = torch.tensor(bg_images, device=self.device, dtype=torch.int16)

        for i in range(n_batches):
            # adjust number of points if they dont fit off_time and density
            idx = self.create_batch(frames, ground_truth, offsets,bg_t, idx)
        offsets.append(idx)
        self.save(np.concatenate(ground_truth, axis=0), offsets, frames)

    def save(self, ground_truth, offsets, frames):
        np.save(self.path + f"/ground_truth", ground_truth)
        np.save(self.path + f"/offsets", offsets)
        imwrite(self.path + "/images.tif", frames)


    def create_batch(self, frames, ground_truth, offsets, bg_t, idx):
        if self.coeff < 1:
            cur_arr = self.o_arr[np.where(np.random.binomial(1, self.coeff * 3, self.o_arr.shape[0]))]
        # todo: works only if coeff is adjusted

        # load points to gpu
        arr = torch.tensor(cur_arr, device=self.device, dtype=torch.float32)
        # load complex trace instead of simulating simple one
        ch = self.load_complex_trace(cur_arr.shape[0], self.batch_size, "data/emitter_traces/flickering.pkl")
        #todo: compute overall crlb with chain (sum over chain)

        # ch = self.simulate_simple_trace(self.off_state_prob, cur_arr.shape[0], self.batch_size)
        for i, values in enumerate(ch):
            indices, photons = torch.tensor(values[:, 0], device=self.device), torch.tensor(values[:, 1],
                                                                                            device=self.device)
            # pick random bg_image
            # select one random background image
            bg_index = torch.randint(0, len(bg_t), (1,), device=self.device)
            # update dropout box every 50 frames
            if i % 50 == 0:
                self.dropout_box.update_box()
            # load remaining coords into simulation
            #data augmentation
            if self.mode == "training":
                dropout_indices, relative_indices = self.dropout_box.forward(arr,indices)
                arr_dropout,photons_dropout = arr[dropout_indices],photons[relative_indices]
                # concatenate data intensity and background
                dat = torch.concat(
                    [arr_dropout, photons_dropout.unsqueeze(1), bg_index.repeat(arr_dropout.shape[0], 1)],
                    dim=1)
            # disable for testing purposes
            else:
                dat = torch.concat(
                    [arr[indices], photons.unsqueeze(1), bg_index.repeat(arr[indices].shape[0], 1)],
                    dim=1)

            # feed bg image here and data
            frame = self.sim(dat, bg_t[bg_index[0]])
            frames.append(frame.cpu().squeeze().numpy())
            # save ground truth
            data = dat.cpu().numpy()
            ground_truth.append(data)
            offsets.append(idx)
            idx += data.shape[0]
        return idx