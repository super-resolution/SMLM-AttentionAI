import torch
import numpy as np

import time


td = torch.distributions
# todo: gaussian mixture




class GMMLoss(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        mesh = torch.meshgrid([torch.arange(0, size[0])+.5, torch.arange(0, size[1])+.5])
        self.grid = torch.stack(mesh, 0)[None, :].to("cuda")


    def grid_coords_xy(self,pos_xy):
        """
        Takes N x 2 X H X W Tensor and adds grid coordinates
        :param input:
        :return:
        """

        return pos_xy + self.grid


    def forward(self, output, pos, mask, bg_truth, seperate=False):

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

        cat = td.Categorical(prob_normed.reshape(batch_size, -1))
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


def my_loss(output, targ, indices):
    #todo: add gridget to output
    mesh = torch.meshgrid([torch.arange(0,output.shape[2]),torch.arange(0,output.shape[3])])
    grid = torch.stack(mesh,-1)

    loss = torch.tensor(0.)
    pre = torch.tensor(1. / torch.sqrt(torch.tensor(2.) * torch.pi ** 2.))
    for out,ind in zip(output,indices):
        target = targ[np.where(ind[:,0]==1)]
        cov_x = torch.concat((out[3:4]**2+0.9, torch.zeros_like(out[3:4])), 0)
        cov_y = torch.concat((torch.zeros_like(out[4:5]), out[4:5]**2+0.9),0)
        cov = torch.inverse(torch.permute(torch.concat([cov_x[None,:],cov_y[None,:]],0),(2,3,0,1)))


        out = torch.permute(out,(1,2,0))[None,:]

        a = (out[:,:,:,1:3] + grid - target)

        b=  cov[None,:,:]
        det = torch.det(cov).squeeze()[None,:]
        gaussian = pre*det*(torch.exp(-0.5 *a[:,:,:,None,:] @ b @ a[:,:,:,:,None])).squeeze()
        prob = out[:,:,:,0]/torch.sum(out[:,:,:,0])[None,]
        log_int = torch.sum(prob*gaussian, dim=[-1,-2])
        l =  -torch.log(log_int+0.001)
        loss += torch.sum(1/target.shape[0]*l)

    return loss
