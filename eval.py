import importlib
import os
import hydra
from hydra.utils import get_original_cwd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from utility.dataset import CustomTrianingDataset
from utility.emitters import Emitter
from visualization.visualization import plot_emitter_set
from third_party.decode.models import SigmaMUNet

def search_attention(net,dataloader):
    res = []
    for sample_batch,truth,mask,bg in dataloader:
        x = truth.cpu().numpy()
        d = {}
        for j in range(x.shape[0]):
            for i in range(x[j].shape[0]):
                x1 = (x[j, i, 0:2] * 100).astype(np.int32)
                if x1[0] == 0:
                    break
                x2 = (x[:, :, 0:2] * 100).astype(np.int32)
                frames = np.where(np.logical_and(x2[:,:,0] == x1[0],x2[:,:,1] == x1[1]))
                d[tuple(x1)] = np.array([[f,x[f,i,2]] for f in frames[0]])#frames and number of photons?

        def MHA_injection(self, inp):
            n_frames = 50
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
            # todo: do this for whole dataset and for 100 frames off time
            # todo: show different trainings of AI
            # todo: compare reconstruction to srrf and cs
            # todo: implement rendering with nn uncertainty
            for tup, framset in d.items():
                ons = np.zeros(n_frames)
                ons[framset[:, 0].astype(np.int16)] = framset[:, 1] / np.sum(d[tup][:, 1])
                # ons = [d[tup][i,1] if i in d[tup][:,0] else 0 for i in range(n_frames)]
                for f, ph in framset:
                    # plt.bar(list(range(n_frames)), ons, label="ON state")
                    res.append(np.corrcoef(ons, y[tup[0] // 100 * 60 + tup[1] // 100, :, int(f)])[0][1])
                    # plt.bar(list(range(n_frames)), y[tup[0]//100 * 60 + tup[1]//100, int(f)], label="attention",alpha=.5)
                    # plt.legend()
                    # plt.savefig("figures/correlation.svg")
                    # plt.show()

            # residual connection + multihead attention
            return z + inp

        net.mha.mha.forward = MHA_injection.__get__(net.mha.mha)
        net(sample_batch)

    plt.hist(res, bins=20)
    plt.ylabel("n samples")
    plt.xlabel("Pearson correlation")
    plt.savefig("figures/corrv.svg")
    plt.show()
    #mha injection into mha block

    #todo: for all batches

    #todo: get attention weights for pixel
    z=0

def reshape_data(images):
    #add temporal context to additional dimnesion
    dataset = np.zeros((images.shape[0],3,images.shape[1],images.shape[2]))
    dataset[1:,0,:,:] = images[:-1]
    dataset[:,1,:,:] = images
    dataset[:-1,2,:,:] = images[1:]
    return dataset

def full_evaluation(dat, emitter_truth, parameter="p", save_name=""):
    title = {"p": "Probability filter", "sig": "Sigma filter"}
    jac = []
    rmse = []
    for i in range(9):
        if parameter == "p":
            filter = 1.0-.05*i
            t = dat.filter(p=filter)#sig_y=sig_filter,sig_x=sig_filter)
        elif parameter == "sig":
            filter = .45-.03*i
            t = dat.filter(sig_y=filter,sig_x=filter)
        rm,ji = t.compute_jaccard(emitter_truth)
        jac.append([filter,ji])
        rmse.append([filter,rm])
    #todo: write to file instead of
    p = f"figures/"+f"threshold_eval.csv"
    S1=pd.Series(jac)
    S2=pd.Series(rmse)
    if os.path.exists(p):
        df = pd.read_csv(p)
    else:
        df = pd.DataFrame()
    df[save_name+parameter+"_Jaccard"] = S1
    df[save_name+parameter+"_RMSE"] = S2
    df.to_csv(p)
    # jac = np.array(jac)
    # rmse = np.array(rmse)
    # fig,axs = plt.subplots()
    # axs.plot(jac[:,0], jac[:,1],"#378f8f", lw=3)
    # axs.set_title(title[parameter])
    # axs.tick_params(axis='y', labelcolor="#378f8f")
    # axs.set_ylabel('JI', color="#378f8f")
    # axs.set_xlabel("threshold")
    # axs2 = axs.twinx()
    # axs2.plot(rmse[:,0], rmse[:,1],'#ff6150', lw=3)
    # axs2.tick_params(axis='y', labelcolor='#ff6150')
    # axs2.set_ylabel('rmse', color='#ff6150')
    # #axs2.set_ylim([0,max(rmse[:,1])])
    # fig.tight_layout()
    # plt.savefig(f"figures/"+save_name+f"threshold_eval_{parameter}.png")
    # plt.clf()
    #plt.show()



@hydra.main(config_name="eval.yaml", config_path="cfg")
def myapp(cfg):
    device = cfg.network.device
    cwd = get_original_cwd()

    dataset_name = cfg.dataset.name
    dataset_offset = cfg.dataset.offset
    #todo: set three channel true if decode
    three_ch = "decode" in cfg.training.name.lower()
    datasets = CustomTrianingDataset(cfg.dataset.name, cwd, three_ch=three_ch, offset=cfg.dataset.offset)
    dataloader = DataLoader(datasets, batch_size=cfg.dataset.batch_size,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=False)

    dtype = getattr(torch, cfg.network.dtype)
    #todo: use dataloader

    model_path = 'trainings/model_'+cfg.training.name#change also in eval
    print(model_path)
    if three_ch:
        net = SigmaMUNet(3)
    else:
        vit = importlib.import_module("models.VIT." + cfg.network.name.lower())  # test if this works
        net = vit.Network(cfg.network.components)

    opt_cls = getattr(torch.optim, cfg.optimizer.name)
    opt = opt_cls(net.parameters(), **cfg.optimizer.params)

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    if cfg.search_attention:
        search_attention(net,dataloader)
    #opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(loss)
    out_data = []
    #evaluation mode
    net.eval()
    #search_attention(net,dataloader)
    truth_list = []
    emitter_truth = None
    for images, truth, mask, bg in dataloader:
        truth_list.append(np.concatenate(truth.cpu().numpy(),axis=0))
        with torch.no_grad():
             out_data.append(net(images).cpu())
        if not emitter_truth:
            emitter_truth = Emitter.from_ground_truth(truth.cpu().numpy())
        else:
            emitter_truth + Emitter.from_ground_truth(truth.cpu().numpy())
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
    dat = Emitter.from_result_tensor(out_data[:, (0,2,3,5,6,7,8,9)], .7,) #maps=net.activation.mapping)#
    #
    #automatically compute the best values
    #dat = dat.filter(sig_y=0.45,sig_x=0.45)
    #todo: update computation and add crlb
    #todo: optimize jaccard:
    full_evaluation(dat, emitter_truth, parameter="sig", save_name=dataset_name+cfg.training.name)
    full_evaluation(dat, emitter_truth, save_name=dataset_name+cfg.training.name)
    dat.compute_jaccard(emitter_truth, "figures/density/"+dataset_name+cfg.training.name, np.concatenate([im.cpu().numpy() for im,_,_,_ in dataloader],axis=0))

    #print(validate(dat,t))
    #plt.plot(jac)
    #plt.savefig("eval_jaccard.svg")
    #plt.show()
    #plot_emitter_gmm(dat)
    plot_emitter_set(dat, save_name="figures/density/"+dataset_name+cfg.training.name)

if __name__ == '__main__':
    myapp()
