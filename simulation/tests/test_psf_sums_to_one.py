import unittest
from simulation.src.data_augmentation import DropoutBox
from copy import deepcopy
import torch
import numpy as np


class TestPsf(unittest.TestCase):
    def setUp(self) -> None:
        # Instantiate the DropoutBox
        self.dropout_box = DropoutBox(1, device="cpu")
        grid = np.arange(0, 60, 1)
        X, Y = np.meshgrid(grid, grid)
        self.coord = torch.tensor(np.moveaxis(np.array([X, Y]).astype(np.float32), 0, -1))
        cov = torch.tensor([[1.8**2,0.],[0,1.8**2.]])

        self.psf = lambda mu,I: torch.sum(I / torch.sqrt((2. * torch.pi)**2 * torch.det(cov)).squeeze() * \
                  torch.exp(-0.5 * torch.sum((self.coord.unsqueeze(2) - mu) @ torch.inverse(cov) * (self.coord.unsqueeze(2) - mu),dim=-1)  ),dim=-1)

    def test_sum(self):
        pdf = self.psf(torch.tensor([[30,30],[31,31],[20,20]]),torch.tensor([2,1,7])).numpy()
        print(pdf.sum(axis=0).sum(axis=0))
        assert pdf.sum()==3

if __name__ == '__main__':
    unittest.main()
