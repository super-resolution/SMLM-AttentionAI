import torch
from torch import Tensor
from collections.abc import Iterable
import numpy as np

import time

td = torch.distributions


class GMMLoss(torch.nn.Module):
    """
    Gaussian Mixture Model loss
    Adapted from Decode
    """
    def __init__(self, size:Iterable[Tensor,np.ndarray]) -> Tensor:
        """
        Initialize a grid for loss
        :param size: Size of output featuremaps/input image
        """
        super().__init__()
        mesh = torch.meshgrid([torch.arange(0, size[0])+.5, torch.arange(0, size[1])+.5])
        self.grid = torch.stack(mesh, 0)[None, :].to("cuda")


    def grid_coords_xy(self,pos_xy:Tensor) -> Tensor:
        """
        Takes N x 2 X H X W Tensor and adds grid coordinates
        :param input: Relative input position to the center of a pixel
        :return: The absolute position in the feature map
        """
        return pos_xy + self.grid


    def forward(self, output:Tensor, pos:Tensor, mask:Tensor, bg_truth:Tensor, seperate=False) -> Iterable[Tensor,list[Tensor]]:
        """
        Forward pass
        :param output: Output of neural network with 8 feature maps
        :param pos: Positions of the ground truth
        :param mask: mask for positions
        :param bg_truth: Background image from simulations
        :param seperate: Give back partial or total loss
        :return: Scalar loss if seperate=False else list of partial tensors
        """
        torch.cuda.synchronize()

        prob = output[:, 0]
        p_xy = output[:, 1:3]
        p_sig = output[:, 3:5]
        bg = output[:, 5]
        N = output[:, 6:7]
        N_sig = output[:, 7:8]


        batch_size = p_xy.shape[0]
        p_xy = self.grid_coords_xy(p_xy)
        p_xy = torch.permute(p_xy, (0, 2, 3, 1)).reshape(batch_size, -1, 2)
        # test = p_xy.masked_select(torch.tensor(mask)[:,:,:,None]).reshape((-1,2))
        N = torch.permute(N, (0, 2, 3, 1)).reshape(batch_size, -1, 1)+0.01
        N_sig = torch.permute(N_sig, (0, 2, 3, 1)).reshape(batch_size, -1, 1)

        p_sig = torch.permute(p_sig, (0, 2, 3, 1)).reshape(batch_size, -1, 2)
        prob_normed = prob / torch.sum(prob, dim=[-1, -2], keepdim=True)

        #p_inds = tuple((prob+1).nonzero(as_tuple=False).transpose(1, 0))
        #p_xy = p_xy[p_inds[0], :, p_inds[1], p_inds[2]]



        n = mask.sum(-1)
        p_mean = torch.sum(prob, dim=[-1, -2])
        p_var = torch.sum((prob - prob ** 2), dim=[-1,-2])
        # var estimate of bernoulli
        p_gauss = td.Normal(p_mean, torch.sqrt(p_var))

        c_loss = torch.sum(-p_gauss.log_prob(n)*n)/100

        #do not validate args. This can lead to simplex error due to rounding
        cat = td.Categorical(prob_normed.reshape(batch_size, -1), validate_args = False)
        comp = td.Independent(td.Normal(p_xy, p_sig), 1)
        gmm = td.mixture_same_family.MixtureSameFamily(cat, comp)
        truth = pos.reshape((batch_size, -1, 3))[:,:,0:2]

        gmm_loss = -gmm.log_prob(truth.transpose(0, 1)).transpose(0, 1)
        gmm_loss = torch.sum(gmm_loss * mask)
        bg_loss = torch.nn.MSELoss()(bg,bg_truth)*10
        if seperate:
            return torch.tensor([gmm_loss,c_loss, bg_loss])
        loss = gmm_loss+c_loss + bg_loss

        return loss


class GMMLossDecode(torch.nn.Module):
    """
    Gaussian Mixture Model loss
    Adapted from Decode
    """
    def __init__(self, size:Iterable[Tensor,np.ndarray]) -> Tensor:
        """
        Initialize a grid for loss
        :param size: Size of output featuremaps/input image
        """
        super().__init__()
        mesh = torch.meshgrid([torch.arange(0, size[0])+.5, torch.arange(0, size[1])+.5])
        self.grid = torch.stack(mesh, 0)[None, :].to("cuda")


    def grid_coords_xy(self,pos_xy:Tensor) -> Tensor:
        """
        Takes N x 2 X H X W Tensor and adds grid coordinates
        :param input: Relative input position to the center of a pixel
        :return: The absolute position in the feature map
        """
        return pos_xy + self.grid


    def forward(self, output:Tensor, pos:Tensor, mask:Tensor, bg_truth:Tensor, seperate=False) -> Iterable[Tensor,list[Tensor]]:
        """
        Forward pass
        :param output: Output of neural network with 8 feature maps
        :param pos: Positions of the ground truth
        :param mask: mask for positions
        :param bg_truth: Background image from simulations
        :param seperate: Give back partial or total loss
        :return: Scalar loss if seperate=False else list of partial tensors
        """
        torch.cuda.synchronize()
        #todo print min max values?
        prob = output[:, 0]
        p_xy = output[:, 2:4]
        p_sig = output[:, 5:7]
        bg = output[:, 9]
        N = output[:, 7:8]
        N_sig = output[:, 8:9]


        batch_size = p_xy.shape[0]
        p_xy = self.grid_coords_xy(p_xy)
        p_xy = torch.permute(p_xy, (0, 2, 3, 1)).reshape(batch_size, -1, 2)
        # test = p_xy.masked_select(torch.tensor(mask)[:,:,:,None]).reshape((-1,2))
        N = torch.permute(N, (0, 2, 3, 1)).reshape(batch_size, -1, 1)+0.01
        N_sig = torch.permute(N_sig, (0, 2, 3, 1)).reshape(batch_size, -1, 1)

        p_sig = torch.permute(p_sig, (0, 2, 3, 1)).reshape(batch_size, -1, 2)
        prob_normed = prob / torch.sum(prob, dim=[-1, -2], keepdim=True)

        #p_inds = tuple((prob+1).nonzero(as_tuple=False).transpose(1, 0))
        #p_xy = p_xy[p_inds[0], :, p_inds[1], p_inds[2]]



        n = mask.sum(-1)
        p_mean = torch.sum(prob, dim=[-1, -2])
        p_var = torch.sum((prob - prob ** 2), dim=[-1,-2])
        # var estimate of bernoulli
        p_gauss = td.Normal(p_mean, torch.sqrt(p_var))

        c_loss = torch.sum(-p_gauss.log_prob(n)*n)/100

        #do not validate args. This can lead to simplex error due to rounding
        cat = td.Categorical(prob_normed.reshape(batch_size, -1), validate_args = False)
        comp = td.Independent(td.Normal(p_xy, p_sig), 1)
        gmm = td.mixture_same_family.MixtureSameFamily(cat, comp)
        truth = pos.reshape((batch_size, -1, 3))[:,:,0:2]

        gmm_loss = -gmm.log_prob(truth.transpose(0, 1)).transpose(0, 1)
        gmm_loss = torch.sum(gmm_loss * mask)
        bg_loss = torch.nn.MSELoss()(bg,bg_truth)*10
        if seperate:
            return torch.tensor([gmm_loss,c_loss, bg_loss])
        loss = gmm_loss+c_loss + bg_loss

        return loss