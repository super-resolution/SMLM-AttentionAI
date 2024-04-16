
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from simulation.src.noise_simulations import Simulation


MOCK_FLIM = {
    "expected_number_of_photons":300.,
    "exposure_time":20e-3,
    "dark_noise":10.,
    "quantum_efficiency":0.9,
    "gain":10.,
    "redout_noise":0.3,
}



class FunctionalTestDataSimulation():
    def __init__(self):
        self.positions = torch.tensor([[3.,3.],[30.,15.],[15.,40.]])


    def plot_stuff_with_default_params(self):
        sim = Simulation.from_dict(MOCK_FLIM)
        image = sim(self.positions)
        plt.imshow(image)
        plt.show()

    def initialize_from_dict(self):
        sim = Simulation.from_dict(MOCK_FLIM)
        image = sim(self.positions)
        plt.imshow(image)
        plt.show()

    def test_troughput_with_for_loop(self):
        sim = Simulation.from_dict(MOCK_FLIM)
        #422 frames p/s should be more than enough
        for i in tqdm(range(1000)):
            image = sim(self.positions)

if __name__ == '__main__':
    test = FunctionalTestDataSimulation()
    test.test_troughput_with_for_loop()