import numpy as np
import torch
import torch.distributions as distributions
from tqdm import tqdm


class SimpleMarkovModel(torch.nn.Module):
    """
    Implements the forward pass of a simple two state markov chain
    """
    def __init__(self, initial, transition, observation, n_frames=500, device="cpu"):
        """

        :param initial: Initial distribution of states
        :param transition: Transition matrix between states
        :param observation: Observation distribution
        :param n_frames: Number of frames to simulate
        :param device: Device to run code on default: cpu runs faster on cuda
        """
        super().__init__()
        dtype = torch.float32
        self.n_frames = n_frames
        # Number of possible states
        self.initial = torch.tensor(initial, dtype=dtype, device=device)
        # transition probabilities
        self.initial_trans = torch.tensor(transition[1][1], device=device)
        #sample transition probabilities to see if a state change happens within a frame
        self.transition = distributions.Categorical(torch.tensor(transition, device=device))
        #transition matrix computes state from transition
        self.transition_matrix = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=dtype, device=device)
        self.observation = torch.tensor(observation, dtype=dtype, device=device)

    def forward(self, change_probabilities=False):
        """
        Apply sampling by binning states into time frames
        Compute whether the state of an emitter changes by applying a markov chain
        :return: Indices of emitter in on state per frame
        """
        state = self.initial
        chain = []
        for i in tqdm(range(self.n_frames)):
            # reduce min
            # update state and cummulative time
            if change_probabilities and i % 100 == 0:
                # change transition probability
                # dont do this if 250er batches are simulated
                p = self.initial_trans * torch.randint(low=1, high=10, size=(1,), device="cuda").type(torch.float32)
                self.transition = distributions.Categorical(torch.tensor([[0.5, 0.5], [1. - p, p]], device="cuda"))
            #sample transitions in the shape of number of points
            A = self.transition.sample([state.shape[0]])
            #Apply state to transition i.e. select row(transitions) of current state
            t = torch.sum(state * A, dim=-1).int()
            #Select corresponding entry in transition matrix to the sampled transition
            m = self.transition_matrix[t]
            #Compute new state
            state = (state.unsqueeze(1) @ m).squeeze()
            #Append all emitters in on state to the current frame
            chain.append(np.where(state.cpu().numpy()[:, 0] == 1)[0])
        return chain


if __name__ == '__main__':
    n_points = 10000
    initial_distribution = torch.tensor([[1, 0]] * n_points, dtype=torch.int8)

    # Emitters have 50% chance to switch into an off state and 0.999 chance to stay there

    transition_distribution = torch.tensor([[0.5, 0.5],
                                            [0.99, 0.01]])

    # final obeservation as joint distribution
    observation_distribution = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

    # We can combine these distributions into a hidden Markov model with n= 30 frames:

    chain = SimpleMarkovModel(initial_distribution, transition_distribution, observation_distribution)
    ch = chain()
    indices = np.array(ch, dtype=np.int8)
    print(indices)
