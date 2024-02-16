import torch
from torch import nn
from torch.distributions import Categorical,Exponential
import numpy as np
from tqdm import tqdm


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
        r_ic = 7*10**8
        r_isc_s0 = 10**4
        r_s1_off = 10**7
        r_off_s0 = 2*10**-2
        rate_matrix = [[0,r_exc,0,0], #S0
                       [r_f,0,r_isc,r_s1_off],   #S1#todo: add ic later
                       [r_isc_s0,0,0,0],   #Trp
                       [r_off_s0,0,0,0],   #Off
                       ]
        #take random choice with probabilities
        #categorical is normed cummulativ row sum
        cumsum = torch.tensor([rates/np.sum(rates) for rates in rate_matrix], device="cuda")
        self.choice_distributions = Categorical(cumsum.to("cuda"))
        #exponential rate is sum of row constants
        self.transition_times = Exponential(rate=torch.tensor([np.sum(rates) for rates in rate_matrix]).to("cuda"))

    def forward(self, n_sample=8000):
        state = torch.zeros(n_sample, dtype=torch.int64)[:,None].to("cuda")
        cum_time = torch.zeros(n_sample)[:,None].to("cuda")
        ph = torch.zeros(n_sample).to("cuda")
        n = torch.tensor([n_sample]).to("cuda")
        for i in tqdm(range(10**5)):
            curstate = torch.gather(self.choice_distributions.sample(n),1,state)
            cum_time += torch.gather(self.transition_times.sample(n),1,state)
            ph = torch.where(torch.logical_and(state==1,curstate==0),ph+1,ph)
            state = curstate
        print(ph)



if __name__ == '__main__':
    s = Simulation()
    s.to("cuda")
    s.forward()