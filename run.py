import hydra
import matplotlib.pyplot as plt
import torch
from tifffile.tifffile import imread
import numpy as np
from utility.emitters import Emitter
from models.VIT.vitv6 import ViT
from visualization.visualization import plot_emitter_set
from tqdm import tqdm

@hydra.main(config_name="eval.yaml", config_path="cfg")
def myapp(cfg):
    device = cfg.network.device
    dataset_name = cfg.dataset.name
    dataset_offset = cfg.dataset.offset
    dtype = getattr(torch, cfg.network.dtype)

    images = []
    files = ["COS7_Phalloidin_ATTO647_1_200_2perHQ_1.tif","COS7_Phalloidin_ATTO647_1_200_2perHQ_1_X2.tif","COS7_Phalloidin_ATTO647_1_200_2perHQ_1_X3.tif"]
    path = r"D:\Daten\Patrick\STORMHD\647" + "\\"
    #path = r"D:\Daten\Danush\STORMHD" + "\\"
    # files = ["NUP107_nb_biotin_StrepAtto513_Biot_Biot_StrepAtto513_60000fr_15msek_200mW_Gain1_MEA100mM_pH_7_0_1proz_3.tif","NUP107_nb_biotin_StrepAtto513_Biot_Biot_StrepAtto513_60000fr_15msek_200mW_Gain1_MEA100mM_pH_7_0_1proz_3_X2.tif","NUP107_nb_biotin_StrepAtto513_Biot_Biot_StrepAtto513_60000fr_15msek_200mW_Gain1_MEA100mM_pH_7_0_1proz_3_X3.tif","NUP107_nb_biotin_StrepAtto513_Biot_Biot_StrepAtto513_60000fr_15msek_200mW_Gain1_MEA100mM_pH_7_0_1proz_3_X4.tif"]
    #path = r"D:\Daten\Rick" + "\\"
    path = r"D:\Daten\Janna\SRRF"+"\\"
    files = ["Munc13-CF568_15ms_17.79real_2000frames_LaserWF_10%LP.tif"]
    #files = ["Gatta94R_20ms_15000fr_Epi_EMCCD_f7,5_256x256px_gain200_quad_line-PFS.tif"]
    for f in files:
        im = imread(path+f)[0:].astype(np.int32)
        im -= im.min()
        #im = np.mean(np.array([im[i*3:(i+1)*3] for i in range(im.shape[0]//3)]),axis=1)/2
        images.append(im)
    images = np.concatenate(images,axis=0)

    #reshape for temporal context
    images = torch.tensor(images, dtype=dtype, device=device)

    model_path = 'trainings/model_'+cfg.training.name
    print(model_path)

    net = ViT(cfg.network.components)
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
    out_data = []

    for i in tqdm(range(images.shape[0]//250)):
        for k in range(1):
            for l in range(1):
                im = images[i*250:(i+1)*250,120*k:120*(k+1),120*l:120*(l+1)]
                with torch.no_grad():
                    out_data.append(net(im).cpu().numpy())
        # out_data = np.concatenate([np.concatenate(out_data[0:2],axis=3),np.concatenate(out_data[2:],axis=3)],axis=2)
        # if i==0:
        #     dat = Emitter.from_result_tensor(out_data, .8)
        # else:
        #     dat + Emitter.from_result_tensor(out_data, .8)
    out_data = np.concatenate(out_data, axis=0)
    #out_data = np.concatenate([np.concatenate([np.concatenate(out_data[::4],axis=0),np.concatenate(out_data[1::4],axis=0)],axis=3),np.concatenate([np.concatenate(out_data[2::4],axis=0),np.concatenate(out_data[3::4],axis=0)],axis=3)],axis=2)#todo: concatenate stuff
    #plt.imshow(np.mean(out_data[:,0],axis=0))
    #plt.show()
    #todo: save and load stuff
    dat = Emitter.from_result_tensor(out_data[:, ], .8)
    dat.save("tmp.npy")
    #todo: needs opengl rendering
    dat = dat.filter(sig_y=0.5,sig_x=0.5)
    #dat.use_dme_drift_correct()
    plt.show()
    plot_emitter_set(dat)

if __name__ == '__main__':
    myapp()