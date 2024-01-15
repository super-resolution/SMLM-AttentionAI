import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

td = torch.distributions
print(torch.cuda.current_device())

class Simulation(nn.Module):
    ALLOWED_KWARGS = ["expected_number_of_photons","exposure_time","dark_noise",
                      "quantum_efficiency","gain", "redout_noise"]
    def __init__(self, device="cpu"):
        super().__init__()
        device = "cuda"
        grid = np.arange(0, 60, 1)
        X, Y = np.meshgrid(grid, grid)
        self.coord = torch.tensor(np.moveaxis(np.array([X, Y]).astype(np.float32), 0, -1), device=device)
        self.expected_number_of_photons = torch.tensor(400., device=device)
        self.exposure_time = torch.tensor(20e-3, device=device)  # s
        self.dark_noise =torch.tensor([ 30.], device=device)  # pro sekunde
        self.quantum_efficiency =torch.tensor([ 0.9], device=device)
        self.gain = torch.tensor(1., device=device)
        self.redout_noise =torch.tensor([ 0.3], device=device)
        d= torch.tensor(2., device=device)
        sig = torch.tensor(1.5, device=device)
        cov = torch.tensor([[sig**2,0.],[0,sig**2.]], device=device)
        self.psf = lambda mu,I: torch.sum((I / torch.sqrt(2. * torch.pi**d) * torch.det(cov)).squeeze() * \
                  torch.exp(-0.5 * torch.sum((self.coord.unsqueeze(2) - mu) @ torch.inverse(cov) * (self.coord.unsqueeze(2) - mu),dim=-1)  ),dim=-1)


    @classmethod
    def from_dict(cls, camera_dict):
        instance = cls()
        for key,value in camera_dict.items():
            if key in instance.ALLOWED_KWARGS:
                #cast value on float
                setattr(instance, key, torch.tensor(value, dtype=torch.float32, device=cls.device))
            else:
                raise Exception(f"Keyword {key} not in list of allowed kwargs")
        return instance


    def forward(self, positions, bg_t=0):
        """
        Simulate microscope image with given probability density function (pdf)
        default PGN Model
        :param pdf: probability density function
        :return: simulated image
        """

        pdf = self.psf(positions[:,0:2],positions[:,2])#todo: add intensity to psf
        pdf = pdf/torch.max(pdf)
        discretepdf = pdf * self.expected_number_of_photons * self.quantum_efficiency
        discretepdf += self.dark_noise * self.exposure_time +bg_t

        x = td.Poisson(discretepdf+0.00001)
        #α > 0 constraint
        x = td.Gamma(x.sample()+0.00001, 1/self.gain)
        x = td.Normal(x.sample(), self.redout_noise)
        return x.sample()


