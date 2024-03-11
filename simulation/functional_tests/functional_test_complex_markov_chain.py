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
        r_isc = 3*10**6#per second
        r_f = 3*10**8
        r_exc = 10**6
        r_ic = 7*10**7
        r_isc_s0 = 10**5
        r_s1_off = 10**4
        r_off_s0 = 2.*10**-1
        rate_matrix = [[0,r_exc,0,0,0], #S0
                       [r_f,0,r_isc,r_s1_off,r_ic],#S1#todo: add ic later
                       [r_isc_s0,0,0,0,0],   #Trp
                       [r_off_s0,0,0,0,0],   #Off
                       ]

        #take random choice with probabilities
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
    def forward(self, n_sample=80000, events=10**4):
        #todo: longer sequence of traces get them with random start index
        state = torch.zeros((n_sample,4), dtype=torch.int64).to("cuda")
        state[:,0] = 1
        cum_time = torch.zeros(events,n_sample)[:,None].to("cuda")
        ph = torch.zeros(events,n_sample).to("cuda")
        n = torch.tensor([n_sample]).to("cuda")
        frames = defaultdict(dict)
        with tqdm(total=events*200) as pbar:
            i,j = 0,0
            while j < events*200:
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
                if i%events == 0:
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
        with open('data/emitter_traces/flickering.pkl', 'wb') as f:
            pickle.dump(frames, f)



if __name__ == '__main__':
    s = Simulation()
    s.to("cuda")
    s.forward()