import time
import os
import hydra
from hydra.utils import get_original_cwd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from models import loader
from models.loss import GMMLossDecode
from utility.dataset import CustomTrianingDataset





@hydra.main(config_name="train.yaml", config_path="cfg")
def myapp(cfg):
    cwd = get_original_cwd()
    torch.manual_seed(0)
    folder_trainings = os.path.join(cwd,"trainings")
    if not os.path.exists(folder_trainings):
        os.mkdir(folder_trainings)
    device = cfg.network.device
    iterations = cfg.training.iterations

    three_ch = "decode" in cfg.training.name.lower()
    datasets = [CustomTrianingDataset(cf,cwd, offset=cfg.dataset.offset, three_ch=three_ch) for cf in cfg.dataset.train]
    train_dataloaders = [DataLoader(data, batch_size=cfg.dataset.batch_size,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=False) for data in datasets]
    validation_dataset = CustomTrianingDataset(cfg.dataset.validation,cwd, offset=cfg.dataset.offset, three_ch=three_ch)
    #note: loss depends on batch size
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.dataset.batch_size, collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=False)

    model_path = os.path.join(folder_trainings, 'model_{}'.format(cfg.training.base))
    save_path = os.path.join(folder_trainings, 'model_{}'.format(cfg.training.name))
    net,opt,loss,epoch = loader.load(model_path, cfg.optimizer, cfg.network, device, decode=three_ch)

    lossf = GMMLossDecode((cfg.dataset.height,cfg.dataset.width))
    lossf.to(device)
    loss_list = [] + loss if loss else []
    #track training time
    t1 = time.time()
    best = 10**9
    for i in range(iterations):
        for train_dataloader in train_dataloaders:
            for images,truth,mask,bg in train_dataloader:
                # plt.imshow(images[100,:,:].cpu().detach())
                # plt.scatter(truth[100,:,1].cpu().detach(), truth[100,:,0].cpu().detach(),c="r")
                # plt.show()

                #set to none speeds up training because gradients are not deleted from memory
                opt.zero_grad(set_to_none=True)
                #can we use lower precision float for speedup?
                #with torch.autocast(device_type="cuda"):
                #does not work at all nan at various steps
                out = net(images)

                loss = lossf(out, truth[:,:,0:3],mask, bg)
                loss.backward()
                opt.step()
        epoch+=1
        #each save point is 2 epochs
        if i%2 ==0:
            #cast eval to disable dropout
            net.eval()
            with torch.no_grad():
                #only validate first batch
                i=0
                v_loss = torch.zeros((3))
                for im,t,m,bg in validation_dataloader:
                    i+=1
                    v_out = net(im)
                    v_loss += lossf(v_out, t[:,:,0:3], m, bg, seperate=True)
                    print(f"loss: {loss} validation_loss: loc_loss = {v_loss[0]}, c_loss = {v_loss[1]} bg_loss = {v_loss[2]}", i)
                #normalize loss by number of batches in validation
                #todo: normalize over batch_size?
                loss_list.append([loss.cpu().numpy(), v_loss.cpu().numpy()/i])
            #set network back to training mode
            net.train()
            if v_loss[0]<best:
                best = v_loss[0]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss_list,
                    'optimizer_params': cfg.optimizer.params,
                    'training_time': time.time()-t1
                }, save_path)
    #this part is deprecated use compare to plot metrics
    # loss_list = np.array(loss_list)
    # plt.plot(loss_list[:,0],label="loss")
    # plt.plot(loss_list[:,1],label="validation_loss")
    # plt.legend()
    # plt.show()
    # plt.imshow(out[0,0].cpu().detach())
    # plt.show()


if __name__ == '__main__':
    myapp()