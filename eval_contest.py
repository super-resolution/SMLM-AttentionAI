import importlib
import os
import hydra
from hydra.utils import get_original_cwd
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from utility.dataset import CustomTrianingDataset, CustomImageDataset
from utility.emitters import Emitter
from visualization.visualization import plot_emitter_set
from third_party.decode.models import SigmaMUNet

def full_evaluation(dat, emitter_truth, parameter="p", save_name=""):
    title = {"p": "Probability filter", "sig": "Sigma filter"}
    jac = []
    rmse = []
    for i in range(9):
        if parameter == "p":
            filter = 0.7-.05*i
            t = dat.filter(p=filter)#sig_y=sig_filter,sig_x=sig_filter)
        elif parameter == "sig":
            filter = .45-.03*i
            t = dat.filter(sig_y=filter,sig_x=filter)
        rm,ji = t.compute_jaccard(emitter_truth)
        jac.append([filter,ji])
        rmse.append([filter,rm])
    #todo: write to file instead of
    p = f"figures/"+f"threshold_contest.csv"
    S1=pd.Series(jac)
    S2=pd.Series(rmse)
    if os.path.exists(p):
        df = pd.read_csv(p)
    else:
        df = pd.DataFrame()
    df[save_name+parameter+"_Jaccard"] = S1
    df[save_name+parameter+"_RMSE"] = S2
    df.to_csv(p)

@hydra.main(config_name="eval.yaml", config_path="cfg")
def myapp(cfg):
    device = cfg.network.device
    cwd = get_original_cwd()

    dataset_name = "ContestHD"
    gt = Emitter.from_thunderstorm_csv(os.path.join(cwd, "data" , dataset_name , "ground_truth.csv"),contest=True)
    three_ch = "decode" in cfg.training.name.lower()
    datasets = CustomImageDataset(dataset_name,cwd, three_ch=three_ch, offset=cfg.dataset.offset)
    dataloader = DataLoader(datasets, batch_size=cfg.dataset.batch_size,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=False)

    dtype = getattr(torch, cfg.network.dtype)

    model_path = os.path.join(cwd,'trainings','model_'+cfg.training.name)#change also in eval
    print(model_path)
    if three_ch:
        net = SigmaMUNet(3)
    else:
        vit = importlib.import_module("models.VIT."+cfg.network.name.lower())#test if this works
        net = vit.Network(cfg.network.components)

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
    dat = Emitter.from_result_tensor(out_data[:, (0,2,3,5,6,7,8,9)], .55) #maps=net.activation.mapping)#
    #
    #automatically compute the best values
    #dat = dat.filter(sig_y=0.25,sig_x=0.25)
    #todo: update computation and add crlb
    #todo: optimize jaccard:
    print(dat.compute_jaccard(gt, output=cwd+"/figures/"+dataset_name+cfg.training.name+".txt", images=np.concatenate([torch.stack(im,dim=0).cpu().numpy() for im in dataloader],axis=0)))

    full_evaluation(dat, gt, parameter="sig", save_name=dataset_name+cfg.training.name)
    full_evaluation(dat, gt, save_name=dataset_name+cfg.training.name)
    #print(validate(dat,t))
    #plt.plot(jac)
    #plt.savefig("eval_jaccard.svg")
    #plt.show()
    #plot_emitter_gmm(dat)
    #todo: mkdir contest
    plot_emitter_set(dat, save_name=cwd+"/figures/contest/"+dataset_name+cfg.training.name)

if __name__ == '__main__':
    myapp()