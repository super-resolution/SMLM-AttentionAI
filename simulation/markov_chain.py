
import torch
import numpy as np
import pdb
from tqdm import tqdm

import torch.distributions as distributions

class HiddenMarkovModel(torch.nn.Module):
    def __init__(self, initial, transition, observation,n_frames=500, device="cpu"):
        super().__init__()
        dtype = torch.float32
        self.n_frames = n_frames
        # Max number of iteration
        # convergence criteria
        # Number of possible states
        self.initial = torch.tensor(initial, dtype=dtype, device=device)
        # transition probabilities
        self.initial_trans = torch.tensor(transition[1][1], device=device)
        self.transition = distributions.Categorical(torch.tensor(transition, device=device))
        #not transition matrix and transition matrix
        #todo: create real transition matrix with transition rates
        #
        r_isc = 3*10**6#per second
        r_f = 3*10**8
        r_exc = 10**6
        r_ic = 7*10**8
        r_isc_s0 = 10**4
        r_s1_off = 10**7
        r_off_s0 = 2*10**-2
        rate_matrix = [[0,r_exc,0,0], #S0
                       [0,r_f,r_isc,r_s1_off],   #S1#todo: add ic later
                       [0,r_isc_s0,0],   #Trp
                       [0,r_off_s0,0,0],   #Off
                       ]

        self.transition_matrix_new = distributions.Exponential(rate=-1/rate_matrix)

        self.transition_matrix = torch.tensor([[[1,0],[0,1]],[[0,1],[1,0]]], dtype=dtype, device=device)
        self.observation = torch.tensor(observation, dtype=dtype, device=device)

    def forward(self):
        state = self.initial
        chain = []
        for i in tqdm(range(self.n_frames)):
            #todo: sample rate matrix
            # reduce min
            # update state and cummulative time

            #todo: apply sampling by binning states into time frames
            #todo: here we can take an intensity value



            if i %100 == 0:
                #change transition probability
                #dont do this if 250er batches are simulated
                p = self.initial_trans*torch.randint(low=1,high=10,size=(1,),device="cuda").type(torch.float32)
                self.transition = distributions.Categorical(torch.tensor([[0.5,0.5],[1.-p, p]], device="cuda"))

            A = self.transition.sample([state.shape[0]])
            t= torch.sum(state*A, dim=-1).int()
            m = self.transition_matrix[t]
            state = (state.unsqueeze(1)@m).squeeze()
            chain.append(np.where(state.cpu().numpy()[:,0]==1)[0])
        return chain



if __name__ == '__main__':
    n_points = 10000
    initial_distribution = torch.tensor([[1, 0]]*n_points, dtype=torch.int8)

    # Emitters have 50% chance to switch into an off state and 0.999 chance to stay there

    transition_distribution = torch.tensor([[0.5, 0.5],
                                                      [0.99, 0.01 ]])

    # final obeservation as joint distribution
    observation_distribution = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

    # We can combine these distributions into a hidden Markov model with n= 30 frames:


    chain = HiddenMarkovModel(initial_distribution, transition_distribution, observation_distribution)
    ch = chain()
    ch = np.array(ch, dtype=np.int8)
    indices = np.where(ch[:,:,0]==1)
    print(indices)
