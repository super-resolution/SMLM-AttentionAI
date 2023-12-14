import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from lion_pytorch import Lion
from tifffile import imread
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from utility.dataset import CustomImageDataset
from utility.emitters import Emitter
from models.loss import GMMLoss
from models.VIT import ViT


def validate(output, truth):
    #todo: test validation
    pred_set = Emitter.from_result_tensor(output.cpu().detach().numpy(), 0.3)
    truth_set = Emitter.from_ground_truth(truth[1].numpy())
    fn = truth_set - pred_set
    fp = pred_set - truth_set
    tp = pred_set % truth_set
    jac = tp.length / (tp.length + fp.length + fn.length)
    return jac

# def reshape_data(images):
#     #add temporal context to additional dimnesion
#     dataset = np.zeros((images.shape[0],3,images.shape[1],images.shape[2]))
#     dataset[1:,0,:,:] = images[:-1]
#     dataset[:,1,:,:] = images
#     dataset[:-1,2,:,:] = images[1:]
#     return dataset

#todo: network to yaml config
@hydra.main(config_name="trainViT.yaml", config_path="cfg")
def myapp(cfg):
    device = cfg.network.device
    iterations = cfg.training.iterations

    #todo: hack background images for now
    path = "data/random_highpower"
    bg_images = imread(path + "/bg_images.tif")
    bg_images = torch.tensor(bg_images, device=device, dtype=torch.float32)
    #todo: hack end
    dataset = CustomImageDataset(cfg.dataset)
    train_dataloader = DataLoader(dataset, batch_size=250,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=True)
    validation_dataset = CustomImageDataset(cfg.dataset, train=False)
    #todo: loss depends on batch size
    validation_dataloader = DataLoader(validation_dataset, batch_size=250,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=True)

    model_path = 'trainings/model_{}'.format(cfg.network.name)

    net = ViT(cfg.network.components)
    loss = None
    if cfg.optimizer.name == "Lion":#todo:try with adam?
        opt = Lion(net.parameters(), **cfg.optimizer.params)
    else:
        opt_cls = getattr(torch.optim, cfg.optimizer.name)
        opt = opt_cls(net.parameters(), **cfg.optimizer.params)
    from_scratch = False
    try:
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model_state_dict']
        opt_dict  = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("loading defined model is successful")
    except:
        from_scratch = True
        net.to(device)
        print("did not find checkpoint")
        epoch=0
    if not from_scratch:
        try:
            net.load_state_dict(state_dict)
            net.to(device)

            opt.load_state_dict(opt_dict)
        except:
            print("model does not fit try tranfer learning instead")
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

            except:
                assert("something did not work out here")


    lossf = GMMLoss((bg_images.shape[1],bg_images.shape[2]))
    lossf.to(device)
    loss_list = [] + loss if loss else []
    #track training time
    t1 = time.time()
    for i in range(iterations):
        for images,truth,mask in train_dataloader:
            # plt.imshow(images[100,:,:].cpu().detach())
            # plt.scatter(truth[100,:,1].cpu().detach(), truth[100,:,0].cpu().detach(),c="r")
            # plt.show()


            #set to none speeds up training because gradients are not deleted from memory
            opt.zero_grad(set_to_none=True)
            out = net(images)

            loss = lossf(out, truth[:,:,0:3],mask, bg_images[truth[:,0,3].type(torch.int32)])
            loss.backward()
            opt.step()
            epoch+=1
        if i%10 ==0:
            with torch.no_grad():
                #only validate first batch
                i=0
                v_loss = 0
                for im,t,m in validation_dataloader:
                    i+=1
                    v_out = net(im)
                    v_loss += lossf(v_out, t[:,:,0:3], m, torch.zeros_like(im, device=device))
                    #print(validate(v_out, t))
                    print(f"loss: {loss} validation_loss: {v_loss}", i)
                loss_list.append([loss.cpu().numpy(), v_loss.cpu().numpy()/i])

            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss_list,
                'optimizer_params': cfg.optimizer.params,
                'training_time': time.time()-t1
            }, model_path)
    loss_list = np.array(loss_list)
    plt.plot(loss_list[:,0],label="loss")
    plt.plot(loss_list[:,1],label="validation_loss")
    plt.legend()
    plt.show()
    plt.imshow(out[0,0].cpu().detach())
    plt.show()


if __name__ == '__main__':
    myapp()