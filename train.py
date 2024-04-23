import importlib
import time
import os
import hydra
from hydra.utils import get_original_cwd
import torch
from lion_pytorch import Lion
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from models.loss import GMMLossDecode
from third_party.decode.models import SigmaMUNet
from utility.dataset import CustomTrianingDataset


def try_to_load_model(model_path, optimizer_cfg, network_cfg, device, decode=False):
    if decode:
        net = SigmaMUNet(3)
        print("loading Decode")
    else:
        net_package = importlib.import_module("models.VIT." + network_cfg.name.lower())
        net = net_package.Network(network_cfg.components)
        print("loading network {}".format(network_cfg.name))
    loss = None
    epoch=0
    if optimizer_cfg.name == "Lion":
        opt = Lion(net.parameters(), **optimizer_cfg.params)
    else:
        opt_cls = getattr(torch.optim, optimizer_cfg.name)
        opt = opt_cls(net.parameters(), **optimizer_cfg.params)
    from_scratch = False
    try:
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model_state_dict']
        opt_dict  = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Loading defined model is successful")
    except:
        from_scratch = True
        net.to(device)
        print("Did not find checkpoint. Training from scratch")
        epoch=0
    if not from_scratch:
        try:
            net.load_state_dict(state_dict)
            net.to(device)
            opt.load_state_dict(opt_dict)
        except KeyError as e:
            print("model does not fit try tranfer learning instead",e)
            try:
                # 1. load net model dict
                new_model_dict = net.state_dict()
                new_op_state = opt.state_dict()
                # 2. overwrite entries in the existing state dict
                filtered_dict = {k: v for k, v in state_dict.items() if k in new_model_dict}
                filtered_optdict = {k: v for k, v in opt_dict.items() if k in new_op_state}

                new_op_state.update(filtered_optdict)
                new_model_dict.update(filtered_dict)
                # 3. load the new state dict
                net.load_state_dict(new_model_dict)
                net.to(device)
                opt.load_state_dict(new_op_state)

            except KeyError as err:
                raise(KeyError("Transfer learning failed", err))
    return net,opt,loss,epoch


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
    net,opt,loss,epoch = try_to_load_model(model_path, cfg.optimizer, cfg.network, device, decode=three_ch)

    lossf = GMMLossDecode((cfg.dataset.height,cfg.dataset.width))
    lossf.to(device)
    loss_list = [] + loss if loss else []
    #track training time
    t1 = time.time()
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