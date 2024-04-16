import numpy as np
import torch
import torch.nn as nn

td = torch.distributions

class Simulation(nn.Module):
    ALLOWED_KWARGS = ["expected_number_of_photons","exposure_time","dark_noise",
                      "quantum_efficiency","gain", "redout_noise","px_size", "psf_sigma"]
    def __init__(self, device="cuda", grid_size:int=60):
        super().__init__()
        #Create simulation grid
        grid = np.arange(0, grid_size, 1)
        X, Y = np.meshgrid(grid, grid)
        self.coord = torch.tensor(np.moveaxis(np.array([X, Y]).astype(np.float32), 0, -1), device=device)
        #Initialize some simulation paramters
        #Adjust to your microsoft or use configuration yaml
        self.expected_number_of_photons = torch.tensor(400., device=device)
        self.exposure_time = torch.tensor(20e-3, device=device)  # s
        self.dark_noise =torch.tensor([ 30.], device=device)  # pro sekunde
        self.quantum_efficiency =torch.tensor([ 0.9], device=device)
        self.gain = torch.tensor(1., device=device)
        self.redout_noise =torch.tensor([ 0.3], device=device)
        #Create Gaussian PSF
        d= torch.tensor(2., device=device)
        self.sig = torch.tensor(1.5, device=device)
        cov = torch.tensor([[self.sig**2,0.],[0,self.sig**2.]], device=device)
        self.psf = lambda mu,I: torch.sum(I / torch.sqrt((2. * torch.pi)**2 * torch.det(cov)).squeeze() * \
                  torch.exp(-0.5 * torch.sum((self.coord.unsqueeze(2) - mu) @ torch.inverse(cov) * (self.coord.unsqueeze(2) - mu),dim=-1)  ),dim=-1)


    @classmethod
    def from_dict(cls, camera_dict:dict, device="cuda", grid_size:int=60) :
        """
        Create simulation class with a dictionary containing the parameters in allowed args
        :param camera_dict: Dict with the attributes of ALLOWED_KWARGS
        :param device: Computation device
        :return: Instance of class with defined initialization parameters
        """
        #Create an instance of Simulation
        instance = cls(grid_size=grid_size)
        #Set attributes for all dict arguments defined in allowed kwargs
        for key,value in camera_dict.items():
            if key in instance.ALLOWED_KWARGS:
                #cast value on float
                setattr(instance, key, torch.tensor(value, dtype=torch.float32, device=device))
            elif key=="name":
                #pass name of the microscope
                pass
            else:
                #Raise exception for unknwon KWARGS
                raise Exception(f"Keyword {key} not in list of allowed kwargs")
        #create psf instance
        d= torch.tensor(2., device=device)
        #sigma set to 186
        sig = torch.tensor(camera_dict.psf_sigma/camera_dict.px_size, device=device)
        #covariance matrix is diagonal till now
        cov = torch.tensor([[sig**2,0.],[0.,sig**2.]], device=device)
        instance.psf = lambda mu,I: torch.sum(I / torch.sqrt((2. * torch.pi)**2 * torch.det(cov)).squeeze() * \
                  torch.exp(-0.5 * torch.sum((instance.coord.unsqueeze(2) - mu) @ torch.inverse(cov) * (instance.coord.unsqueeze(2) - mu),dim=-1)  ),dim=-1)
        return instance


    def forward(self, positions:torch.tensor, bg_t:torch.tensor=0)->torch.tensor:
        """
        Simulate microscope image with given probability density function (pdf)
        default PGN Model
        :param pdf: probability density function
        :return: Simulated image
        """
        #Create probability density function
        pdf = self.psf(positions[:,0:2],positions[:,2])
        #do not devide by zero
        # m = torch.max(pdf)
        # if m>.0001:
        #     pdf = pdf/torch.max(pdf)
        #Multiply number of photons and quantum efficiency
        #todo: discard expected number of photons
        discretepdf = pdf * self.quantum_efficiency
        x = discretepdf.cpu().numpy()
        discretepdf += self.dark_noise * self.exposure_time +bg_t//2
        #Sample poisson of discrete pdf (Photon shot noise)
        x = td.Poisson(discretepdf+0.00001)
        #Add multiplication noise
        x = td.Gamma(x.sample()+0.00001, 1/self.gain)
        # Add readout noise
        x = td.Normal(x.sample(), self.redout_noise)
        return x.sample()


