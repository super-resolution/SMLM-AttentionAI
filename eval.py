import importlib

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from utility.dataset import CustomImageDataset
from utility.emitters import Emitter
from visualization.visualization import plot_emitter_set
from third_party.decode.models import SigmaMUNet

def search_attention(net,dataloader):
    sample_batch,truth,mask,bg = next(iter(dataloader))
    x = truth.cpu().numpy()
    d = {}
    for i in range(x[0].shape[0]):
        x1 = (x[0, i, 0:2] * 100).astype(np.int32)
        if x1[0] == 0:
            break
        x2 = (x[:, :, 0:2] * 100).astype(np.int32)
        frames = np.where(np.logical_and(x2[:,:,0] == x1[0],x2[:,:,1] == x1[1]))
        d[tuple(x1)] = frames[0]
    #todo: get close
    tup = (1246, 971)
    def MHA_injection(self, inp):
        # Layer norm
        x = self.norm(inp)
        # Compute Query Key and Value
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        z, y = self.mha(q, k, v, need_weights=True)
        y = y.cpu().detach().numpy()
        # todo: also plot close loc
        # this is attention weight of 1
        res = []
        #todo: do this for whole dataset and for 100 frames off time
        #todo: show different trainings of AI
        #todo: compare reconstruction to srrf and cs
        #todo: implement rendering with nn uncertainty
        for tup,framset in d.items():
            for f in framset:
                ons = [.01 if i in d[tup] else 0 for i in range(250)]
                # plt.bar(list(range(250)), ons, label="ON state")
                res.append(np.corrcoef(ons,y[tup[0]//100 * 60 + tup[1]//100, f])[0][1])
                # plt.bar(list(range(250)), y[tup[0]//100 * 60 + tup[1]//100, f], label="attention")
                # plt.legend()
                # plt.savefig("figures/correlation.svg")
                # plt.show()
        plt.hist(res,bins=20)
        plt.ylabel("n samples")
        plt.xlabel("Pearson correlation")
        plt.savefig("figures/corrv.svg")

        plt.show()
        # residual connection + multihead attention
        return z + inp

    net.decoder.mha.forward = MHA_injection.__get__(net.decoder.mha)
    #todo: write inject for testing MHA?

    net(sample_batch)
    #todo: get attention weights for pixel
    z=0

def reshape_data(images):
    #add temporal context to additional dimnesion
    dataset = np.zeros((images.shape[0],3,images.shape[1],images.shape[2]))
    dataset[1:,0,:,:] = images[:-1]
    dataset[:,1,:,:] = images
    dataset[:-1,2,:,:] = images[1:]
    return dataset





@hydra.main(config_name="eval.yaml", config_path="cfg")
def myapp(cfg):
    device = cfg.network.device
    dataset_name = cfg.dataset.name
    dataset_offset = cfg.dataset.offset

    datasets = CustomImageDataset(cfg.dataset.name, three_ch=True, offset=cfg.dataset.offset)
    dataloader = DataLoader(datasets, batch_size=cfg.dataset.batch_size,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=False)

    dtype = getattr(torch, cfg.network.dtype)
    #todo: use dataloader

    # arr = np.load("data/"+ dataset_name + "/coords.npy" , allow_pickle=True)[:,::-1]
    # indices = np.load("data/" + dataset_name + "/indices.npy", allow_pickle=True)[dataset_offset:]
    # truth = []
    # for i, ind in enumerate(indices):
    #     val = arr[np.where(ind[:, 0] != 0)]
    #     truth.append(val)


    #
    #images = imread(r"D:\Daten\Patrick\STORMHD\643\COS7_Phalloidin_ATTO643_1_200_2perHQ_4.tif")[13000:14000,60:130,60:130].astype(np.float32)/24
    #images -= images.min()
    #reshape for temporal context
    #images = torch.nn.functional.pad(images, (0,0,0,1,0,1))
    model_path = 'trainings/model_'+cfg.training.name#change also in eval
    print(model_path)
    vit = importlib.import_module("models.VIT."+cfg.network.name.lower())#test if this works

    net = SigmaMUNet(3)
    #opt_cls = getattr(torch.optim, cfg.optimizer.name)
    #opt = opt_cls(net.parameters(), **cfg.optimizer.params)

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    #opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(loss)
    out_data = []
    #evaluation mode
    net.eval()
    #search_attention(net,dataloader)
    truth_list = []
    for images, truth, mask, bg in dataloader:
        truth_list.append(truth.cpu().numpy())
        with torch.no_grad():
             out_data.append(net(images).cpu())
    out_data = torch.concat(out_data,dim=0)
    truth = np.concatenate(truth_list,axis=0)
    t = Emitter.from_ground_truth(truth)
    out_data = out_data.numpy()
    # plt.imshow(np.mean(out_data[:,0],axis=0),cmap="hot")#todo: plot mean and std
    # plt.colorbar()
    #
    # plt.savefig("figures/avg_p.svg")

    #plt.scatter(truth[0][:,1],truth[0][:,0])
    #plt.show()
    #truth = Emitter.from_ground_truth(truth)
    jac= []
    # for i in range(8):
    dat = Emitter.from_result_tensor(out_data[:, (0,2,3,5,6,7,8,9)], 0.5)
    #
    #dat = dat.filter(sig_y=0.9,sig_x=0.9)
    #dat.compute_jaccard(t, out_data[:,0])

    #print(validate(dat,t))
    #plt.plot(jac)
    #plt.savefig("eval_jaccard.svg")
    #plt.show()
    #plot_emitter_gmm(dat)
    plot_emitter_set(dat)

if __name__ == '__main__':
    myapp()
