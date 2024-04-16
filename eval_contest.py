import importlib

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from utility.dataset import CustomTrianingDataset, CustomImageDataset
from utility.emitters import Emitter
from visualization.visualization import plot_emitter_set
from third_party.decode.models import SigmaMUNet



@hydra.main(config_name="eval.yaml", config_path="cfg")
def myapp(cfg):
    device = cfg.network.device
    dataset_name = "ContestHD"
    gt = Emitter.from_thunderstorm_csv("data/" + dataset_name + "/ground_truth.csv",contest=True)
    three_ch = "decode" in cfg.training.name.lower()
    datasets = CustomImageDataset(dataset_name, three_ch=three_ch, offset=cfg.dataset.offset)
    dataloader = DataLoader(datasets, batch_size=cfg.dataset.batch_size,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=False)

    dtype = getattr(torch, cfg.network.dtype)

    model_path = 'trainings/model_'+cfg.training.name#change also in eval
    print(model_path)
    vit = importlib.import_module("models.VIT."+cfg.network.name.lower())#test if this works
    if three_ch:
        net = SigmaMUNet(3)
    else:
        net = vit.ViT(cfg.network.components)

    opt_cls = getattr(torch.optim, cfg.optimizer.name)
    opt = opt_cls(net.parameters(), **cfg.optimizer.params)

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
    for images in dataloader:
        with torch.no_grad():
            im = torch.stack(images,dim=0)
            out_data.append(net(im).cpu())

    out_data = torch.concat(out_data,dim=0)
    # t = Emitter.from_ground_truth(truth)
    out_data = out_data.numpy()
    #plt.imshow(np.mean(out_data[:,0],axis=0),cmap="hot")#todo: plot mean and std
    #plt.colorbar()

    # plt.savefig("figures/avg_p.svg")

    #plt.scatter(truth[0][:,1],truth[0][:,0])
    #plt.show()
    jac= []
    # for i in range(8):
    #todo: create mapping for output
    dat = Emitter.from_result_tensor(out_data[:, (0,2,3,5,6,7,8,9)], .4,) #maps=net.activation.mapping)#
    #
    #automatically compute the best values
    dat = dat.filter(sig_y=0.45,sig_x=0.45)
    #todo: update computation and add crlb
    #todo: optimize jaccard:
    print(dat.compute_jaccard(gt, ))

    #print(validate(dat,t))
    #plt.plot(jac)
    #plt.savefig("eval_jaccard.svg")
    #plt.show()
    #plot_emitter_gmm(dat)
    plot_emitter_set(dat, save_name="figures/density/"+dataset_name+cfg.training.name)

if __name__ == '__main__':
    myapp()