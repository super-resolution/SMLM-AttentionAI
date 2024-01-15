import unittest

import torch
import torch.distributions as distributions
import tqdm

class ModuleTestDataSimulation(unittest.TestCase):
    def setUp(self) -> None:
        r_isc = 3*10**6#per second
        r_f = 3*10**8
        r_exc = 10**6
        r_ic = 7*10**8
        r_isc_s0 = 10**4
        r_s1_off = 10**7
        r_off_s0 = 2*10**-2
        v = [[0,r_exc,0,0], #S0
                       [r_f,0,r_isc,r_s1_off],   #S1#todo: add ic later
                       [r_isc_s0,0,0,0],   #Trp
                       [r_off_s0,0,0,0],   #Off
                       ]
        v = [[x if x!=0 else 1 for x in y ] for y in v]
        rate_matrix = torch.tensor(v)

        self.transition_matrix_new = [distributions.Exponential(rate=rate) for rate in rate_matrix]
        self.state = 0

    def test_stuff(self):
        tot_time = 0
        while tot_time < 300:
            x = self.transition_matrix_new[self.state].sample()
            ind = torch.argmin(x)#,dim=1
            tot_time += x[ind]
            self.state = ind
            if (tot_time//1)%10==0:
                print(tot_time, self.state)
