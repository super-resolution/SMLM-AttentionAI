import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm
tfd = tfp.distributions

class Simulation():
    def run(self, n_points=5000, frames=100):

        initial_distribution = tfd.Categorical(probs=[[1., .0,.0,.0]] )

        # Emitters have 50% chance to switch into an off state and 0.999 chance to stay there
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
        rate_norm = np.array([rates / np.sum(rates) for rates in rate_matrix])
        cumsum = np.array([np.sum(rates) for rates in rate_matrix])
        transition_distribution = tfd.Categorical(probs=[rate_norm] * 1)

        # final obeservation as joint distribution
        observation_distribution = tfd.Exponential(rate=[cumsum] * 1, name="probs")
        photon_distribution = tfd.Categorical(probs=tf.eye(4, batch_shape=[1]), name="probs")

        # We can combine these distributions into a hidden Markov model with n= 30 frames:
        x = tfd.HiddenMarkovModel(
            initial_distribution=initial_distribution,
            transition_distribution=transition_distribution,
            observation_distribution=photon_distribution,
            num_steps=frames, name="probs"
        )
        for i in tqdm(range(100)):
            x.sample(n_points)

if __name__ == '__main__':
    s = Simulation()
    x = np.array(s.run())
    print(s.run())