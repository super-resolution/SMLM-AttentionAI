import importlib

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile.tifffile import imread
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from utility.dataset import CustomImageDataset
from utility.emitters import Emitter
from utility.visualization import plot_emitter_set


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

    datasets = [CustomImageDataset(cf,  offset=cfg.dataset.offset) for cf in cfg.dataset.name]
    test_dataloaders = [DataLoader(data, batch_size=cfg.dataset.batch_size,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=True) for data in datasets]

    dtype = getattr(torch, cfg.network.dtype)
    #todo: use dataloader

    # arr = np.load("data/"+ dataset_name + "/coords.npy" , allow_pickle=True)[:,::-1]
    # indices = np.load("data/" + dataset_name + "/indices.npy", allow_pickle=True)[dataset_offset:]
    # truth = []
    # for i, ind in enumerate(indices):
    #     val = arr[np.where(ind[:, 0] != 0)]
    #     truth.append(val)


    images = imread("data/"+ dataset_name + "/images.tif")[0:4000,0:60,0:60]
    #
    #images = imread(r"D:\Daten\Patrick\STORMHD\643\COS7_Phalloidin_ATTO643_1_200_2perHQ_4.tif")[13000:14000,60:130,60:130].astype(np.float32)/24
    #images -= images.min()
    #reshape for temporal context
    images = torch.tensor(images, dtype=dtype, device=device)
    #images = torch.nn.functional.pad(images, (0,0,0,1,0,1))
    model_path = 'trainings/model_'+cfg.network.name#change also in eval
    print(model_path)
    vit = importlib.import_module("models.VIT."+cfg.network.name.lower())#test if this works

    net = vit.ViT(cfg.network.components)
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
    for i in range(0,images.shape[0],250):
        with torch.no_grad():
             out_data.append(net(images[i:i+250]).cpu())
    out_data = torch.concat(out_data,dim=0)

    out_data = out_data.numpy()
    plt.imshow(np.min(out_data[:,1],axis=0))#todo: plot mean and std
    #plt.scatter(truth[0][:,1],truth[0][:,0])
    plt.show()
    #truth = Emitter.from_ground_truth(truth)
    jac= []
    # for i in range(8):
    dat = Emitter.from_result_tensor(out_data[:, ], .6)
    #
    #dat = dat.filter(sig_y=0.1,sig_x=0.1)
    #     jac.append(validate(dat, truth))
    #plt.plot(jac)
    #plt.savefig("eval_jaccard.svg")
    #plt.show()
    plot_emitter_set(dat)

if __name__ == '__main__':
    myapp()
