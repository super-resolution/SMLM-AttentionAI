import os
import torch
from torch import nn
from torch.distributions import Categorical,Exponential
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle

class Simulation(nn.Module):
    """
    More complex markov chain model taking rates into account
    Could be used to create emission traces for emitters that can be augmented
    """
    def __init__(self):
        super().__init__()
        #todo: save rate matrix
        r_isc = 3*10**6#per second
        r_f = 5*10**7
        r_exc = 10**6
        r_ic = 7*10**7
        r_isc_s0 = 10**5
        r_s1_off = 2*10**4
        r_off_s0 = 2.*10**-1
        rate_matrix = [[0,r_exc,0,0,0], #S0
                       [r_f,0,r_isc,r_s1_off,r_ic],#S1
                       [r_isc_s0,0,0,0,0],   #Trp
                       [r_off_s0,0,0,0,0],   #Off
                       ]
        #take random choice with probabilities
        self.rate_matrix = np.array(rate_matrix)
        #categorical is normed cummulativ row sum
        self.cumsum = torch.tensor([rates/np.sum(rates) for rates in rate_matrix], device="cuda")
        self.choice_distributions = Categorical(self.cumsum.to("cuda"))
        #exponential rate is sum of row constants
        self.transition_times = Exponential(rate=torch.tensor([np.sum(rates) for rates in rate_matrix]).to("cuda"))
        self.transition_matrix = torch.zeros((5,4),device="cuda")
        for i in range(4):
            self.transition_matrix[i,i] = 1
        self.transition_matrix[4,0] = 1
        self.photon_mat = torch.zeros((5,4),device="cuda")
        self.photon_mat[1,0] = 1
    def forward(self, n_sample=80000, events_p_batch=10**4,batches=2):
        state = torch.zeros((n_sample,4), dtype=torch.int64).to("cuda")
        state[:,0] = 1
        cum_time = torch.zeros(events_p_batch,n_sample)[:,None].to("cuda")
        ph = torch.zeros(events_p_batch,n_sample).to("cuda")
        n = torch.tensor([n_sample]).to("cuda")
        frames = defaultdict(dict)
        with tqdm(total=events_p_batch*batches) as pbar:
            i,j = 0,0
            while j < events_p_batch*batches:
                pbar.update(1)
                #x = self.cumsum[state].squeeze()
                #curstate = torch.multinomial(x, 1, True)
                #is faster despite computational overhead
                transition_m = self.choice_distributions.sample(n)
                t = torch.sum(state * transition_m, dim=-1).int()

                curstate = self.transition_matrix[t]
                ph[i] = self.photon_mat[t,0]

                cum_time[i] = cum_time[i-1]+(self.transition_times.sample(n)*state).sum(dim=1)

                state = curstate
                i += 1
                j += 1
                if i%events_p_batch == 0:
                    #bucketize simulated photons to bin them into a frame
                    bins = torch.arange(0,400,0.002, device="cuda")
                    ind = torch.bucketize(cum_time,bins)
                    #discard first and last bucket
                    v = ind.squeeze()*ph
                    ph_frame = v.cpu().numpy()
                    for k in range(n_sample):
                        z = np.unique(ph_frame[:,k], return_counts=True)
                        # per frame save index, photons
                        for frame_index,count in zip(*z):
                            #hashing the corresponding frames is fast
                            if k in frames[frame_index]:
                                frames[frame_index][k] += count
                            else:
                                frames[frame_index][k] = count
                    i=0
        cwd = os.getcwd()
        folder_data = os.path.join(cwd,"data")
        folder_traces = os.path.join(folder_data,"emitter_traces")
        f_name = "switching"
        if not os.path.exists(folder_data):
            os.mkdir(folder_data)
        if not os.path.exists(folder_traces):
            os.mkdir(folder_traces)
        with open(os.path.join(folder_traces, f_name + '.pkl'), 'wb') as f:
            pickle.dump(frames, f)
        np.savetxt(os.path.join(folder_traces, f_name + ".txt"), self.rate_matrix)



if __name__ == '__main__':
    s = Simulation()
    s.to("cuda")
    s.forward()