import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile.tifffile import imread
from compare_metrics import validate

from emitters import Emitter
from network import Network2,AttentionIsAllYouNeed,RecursiveUNet, FFTAttentionUNet,Network3
from visualization import plot_emitter_set
from models.VIT import ViT

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


    dtype = getattr(torch, cfg.network.dtype)
    arr = np.load("data/"+ dataset_name + "/coords.npy" , allow_pickle=True)[:,::-1]
    indices = np.load("data/" + dataset_name + "/indices.npy", allow_pickle=True)[dataset_offset:]
    truth = []
    for i, ind in enumerate(indices):
        val = arr[np.where(ind[:, 0] != 0)]
        truth.append(val)


    images = imread("data/"+ dataset_name + "/images.tif")[dataset_offset:3000,0:60,0:60]*2
    # images = imread(r"D:\Daten\Patrick\STORMHD\643\COS7_Phalloidin_ATTO643_1_200_2perHQ_4.tif")[10000:15000,60:120,60:120].astype(np.float32)/12
    #images -= images.min()
    #reshape for temporal context
    images = torch.tensor(images, dtype=dtype, device=device)
    images = torch.nn.functional.pad(images, (0,0,0,1,0,1))
    model_path = 'trainings/model_ViTV2'
    print(model_path)

    net = ViT()
    #opt_cls = getattr(torch.optim, cfg.optimizer.name)
    #opt = opt_cls(net.parameters(), **cfg.optimizer.params)

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    #opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(loss)

    net.eval()
    with torch.no_grad():
        out_data = net(images).cpu()

    out_data = out_data.numpy()
    plt.imshow(out_data[0,0])
    plt.scatter(truth[0][:,1],truth[0][:,0])
    plt.show()
    #truth = Emitter.from_ground_truth(truth)
    jac= []
    # for i in range(8):
    dat = Emitter.from_result_tensor(out_data[:, ], .5)
    #
    #dat = dat.filter(sig_y=0.3,sig_x=0.3)
    #     jac.append(validate(dat, truth))
    #plt.plot(jac)
    #plt.savefig("eval_jaccard.svg")
    #plt.show()
    plot_emitter_set(dat)

if __name__ == '__main__':
    myapp()
