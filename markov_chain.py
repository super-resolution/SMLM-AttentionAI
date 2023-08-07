
import torch
import numpy as np
import pdb
from tqdm import tqdm

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
        self.transition = torch.distributions.Categorical(torch.tensor(transition, device=device))
        #not transition matrix and transition matrix
        self.transition_matrix = torch.tensor([[[1,0],[0,1]],[[0,1],[1,0]]], dtype=dtype, device=device)
        self.observation = torch.tensor(observation, dtype=dtype, device=device)

    def forward(self):
        state = self.initial
        chain = []
        for i in tqdm(range(self.n_frames)):
            if i% 500 ==0:
                p = np.random.rand(1)*0.001
                self.transition = torch.distributions.Categorical(torch.tensor([[0.8,0.2],[1.-p, p]], device="cuda"))

            A = self.transition.sample([state.shape[0]])
            t= torch.sum(state*A, dim=-1).int()
            m = self.transition_matrix[t]
            state = (state.unsqueeze(1)@m).squeeze()
            chain.append(state.cpu())
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
