import unittest

import matplotlib.pyplot as plt
import torch

from noise_simulations import Simulation

#todo: dict to hydra conf
MOCK_FLIM = {
    "expected_number_of_photons":300.,
    "exposure_time":20e-3,
    "dark_noise":10.,
    "quantum_efficiency":0.9,
    "gain":10.,
    "redout_noise":0.3,
}


class ModuleTestDataSimulation(unittest.TestCase):
    def setUp(self):
        #initialize with default parameters
        self.sim = Simulation()
        self.positions = torch.tensor([[3.,3.],[30.,15.],[15.,40.]])

    def test_initialize_from_dict(self):
        sim = Simulation.from_dict(MOCK_FLIM)
        image = sim(self.positions)
        plt.imshow(image.sample())
        plt.show()

