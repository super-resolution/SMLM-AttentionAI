import unittest
import torch
import numpy as np
td = torch.distributions

class CustomnGMM(torch.nn.Module):
    def forward(self):
        return 0


class TestMaskedLoss(unittest.TestCase):
    def setUp(self) -> None:
        #todo: create ground truth tensor
        data = np.zeros((10,20,2))
        probs = np.zeros((10,20))
        self.ns = []
        for i in range(10):
            n = np.random.randint(1, 20)
            self.ns.append(n)
            for j in range(n):
                probs[i,j] = 1/n
                #random coordinates in the range [0,10]
                data[i,j] = np.random.rand(2)*10
        self.data = torch.tensor(data)
        self.probs = torch.tensor(probs)
        sig = torch.ones_like(self.data)
        cat = td.Categorical(self.probs)
        comp = td.Independent(td.Normal(self.data, sig), 1)
        self.gmm = td.mixture_same_family.MixtureSameFamily(cat, comp)

        #create GMM model

    def test_log_prob_masked_tensor(self):
        print(self.gmm.log_prob(self.data.transpose(0, 1)))
        masked = torch.masked.MaskedTensor(self.data, self.probs.to(torch.bool)[:,:,None].repeat(1,1,2))
        loss = 0
        for i,n in enumerate(self.ns):
            loss += self.gmm.log_prob(self.data[i,0:n,None])
        print(loss)


if __name__ == '__main__':
    unittest.main()