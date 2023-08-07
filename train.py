import torch
import hydra
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from dataset import CustomImageDataset
from loss import GMMLoss
from network import Network2, Activation
from lion_pytorch import Lion

# def reshape_data(images):
#     #add temporal context to additional dimnesion
#     dataset = np.zeros((images.shape[0],3,images.shape[1],images.shape[2]))
#     dataset[1:,0,:,:] = images[:-1]
#     dataset[:,1,:,:] = images
#     dataset[:-1,2,:,:] = images[1:]
#     return dataset

#todo: network to yaml config
@hydra.main(config_name="train.yaml", config_path="cfg")
def myapp(cfg):
    device = cfg.network.device
    dataset_name = cfg.dataset.train
    dataset_offset = cfg.dataset.offset
    iterations = cfg.training.iterations
    dtype = getattr(torch, cfg.network.dtype)

    validation_dataset = CustomImageDataset(cfg.dataset, train=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=256,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=True)


    dataset = CustomImageDataset(cfg.dataset)
    train_dataloader = DataLoader(dataset, batch_size=128,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=True)


    model_path = 'trainings/model_{}'.format(cfg.network.name)

    net = Network2()
    if cfg.optimizer.name == "Lion":
        opt = Lion(net.parameters(), **cfg.optimizer.params)
    else:
        opt_cls = getattr(torch.optim, cfg.optimizer.name)
        opt = opt_cls(net.parameters(), **cfg.optimizer.params)

    try:
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)

        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(loss)
    except:
        print("did not find checkpoint")
        epoch=0
        net.to(device)


    lossf = GMMLoss((60,60))
    lossf.to(device)
    loss_list = []
    for i in range(iterations):
        for images,truth,mask in train_dataloader:
            # plt.imshow(images[100,:,:].cpu().detach())
            # plt.scatter(truth[100,:,1].cpu().detach(), truth[100,:,0].cpu().detach(),c="r")
            # plt.show()


            #set to none speeds up training because gradients are not deleted from memory
            opt.zero_grad(set_to_none=True)
            out = net(images)

            loss = lossf(out, truth,mask)
            loss.backward()
            opt.step()
            epoch+=1
            if i%10 ==0:
                with torch.no_grad():
                    #only validate first batch
                    im,t,m = next(iter(validation_dataloader))
                    v_out = net(im)
                    v_loss = lossf(v_out, t, m)
                    print(f"loss: {loss} validation_loss: {v_loss}")
                    loss_list.append(loss)

    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss_list,
        'optimizer_params': cfg.optimizer.params
    }, model_path)

    plt.imshow(out[0,0].cpu().detach())
    plt.show()


if __name__ == '__main__':
    myapp()