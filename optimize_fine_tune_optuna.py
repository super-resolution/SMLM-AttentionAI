import random
import time
import os
import hydra
from hydra.utils import get_original_cwd
import torch
import optuna
from sklearn.model_selection  import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from utility.emitters import Emitter
import logging
from itertools import chain
from models import loader
from models.loss import GMMLossDecode
from utility.dataset import CustomTrianingDataset
import numpy as np




@hydra.main(config_name="train.yaml", config_path="cfg")
def myapp(cfg):
    cwd = get_original_cwd()
    torch.manual_seed(0)
    folder_trainings = os.path.join(cwd,"trainings")
    if not os.path.exists(folder_trainings):
        os.mkdir(folder_trainings)
    device = cfg.network.device

    three_ch = "decode" in cfg.training.name.lower()
    datasets = [CustomTrianingDataset(cf,cwd, offset=cfg.dataset.offset, three_ch=three_ch) for cf in cfg.dataset.train]
    train_dataloaders = [DataLoader(data, batch_size=cfg.dataset.batch_size,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=False) for data in datasets]


    model_path = os.path.join(folder_trainings, 'model_{}'.format(cfg.training.base))
    save_path = os.path.join(folder_trainings, 'model_{}'.format(cfg.training.name))

    o_best = [0]
    def train(trial, device="cuda"):

        # todo: enum for norm in network
        # todo: loguniform for learning rate
        # todo: uniform for hidden_d
        # todo: uniform for positional encoding
        optimizer_cfg = {'name': trial.suggest_categorical('name', ['AdamW']),
                         "params": {"lr": trial.suggest_loguniform('learning_rate', 10 ** (-5), 10 ** (-3))}
                         }
        #todo only vary one multiplier?
        multiplier = [trial.suggest_float("loc_multiplier", 1,1), trial.suggest_float("count_multiplier", 1.,10), trial.suggest_float("bg_multiplier", 0.1,0.5)]
        net, opt, loss, epoch = loader.load(model_path, optimizer_cfg, cfg.network, device)

        lossf = GMMLossDecode((cfg.dataset.height, cfg.dataset.width),multiplier)
        lossf.to(device)
        loss_list = [] + loss if loss else []
        # track training time
        t1 = time.time()
        iterations = cfg.training.iterations
        n = len(datasets[0])//cfg.dataset.batch_size
        picks = set(list(range(n)))
        test_batches = random.sample(list(picks), int(n*0.2))
        train_batches = list(picks-set(test_batches))
        train_idx = []
        for id in train_batches:
            train_idx += list(range(id*cfg.dataset.batch_size, (id+1)*cfg.dataset.batch_size))
        test_idx = []
        for id in test_batches:
            test_idx += list(range(id*cfg.dataset.batch_size, (id+1)*cfg.dataset.batch_size))
        train_dataset = torch.utils.data.Subset(datasets[0], train_idx)
        test_dataset = torch.utils.data.Subset(datasets[0], test_idx)
        train = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size,
                   collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)),
                   shuffle=False)
        test = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size,
                   collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)),
                   shuffle=False)
        best = 0
        for i in range(iterations):
            for images, truth, mask, bg in train:
                # plt.imshow(images[100,:,:].cpu().detach())
                # plt.scatter(truth[100,:,1].cpu().detach(), truth[100,:,0].cpu().detach(),c="r")
                # plt.show()

                # set to none speeds up training because gradients are not deleted from memory
                opt.zero_grad(set_to_none=True)
                out = net(images)

                loss = lossf(out, truth[:, :, 0:3], mask, bg)
                loss.backward()
                opt.step()
            epoch += 1
            # each save point is 2 epochs
            if i % 2 == 0:
                # cast eval to disable dropout
                net.eval()
                with torch.no_grad():
                    # only validate first batch
                    i = 0
                    v_loss = torch.zeros((3))
                    out_data = []
                    emitter_truth = None
                    for im, t, m, bg in test:
                        # todo: validate with JI and rmse
                        if not emitter_truth:
                            emitter_truth = Emitter.from_ground_truth(t.cpu().numpy())
                        else:
                            emitter_truth + Emitter.from_ground_truth(t.cpu().numpy())
                        i += 1
                        v_out = net(im)
                        v_loss += lossf(v_out, t[:, :, 0:3], m, bg, seperate=True)
                        # print(
                        #     f"loss: {loss} validation_loss: loc_loss = {v_loss[0]}, c_loss = {v_loss[1]} bg_loss = {v_loss[2]}",
                        #     i)
                        out_data.append(v_out.cpu())
                    out_data = torch.concat(out_data, dim=0)
                    # t = Emitter.from_ground_truth(truth)
                    out_data = out_data.numpy()
                    try:
                        dat = Emitter.from_result_tensor(out_data[:, (0, 2, 3, 5, 6, 7, 8, 9)], .7, )
                        rmse, ji = dat.compute_jaccard(emitter_truth)
                        E = 100 - np.sqrt((100 - ji * 100) ** 2 + rmse ** 2)
                        print("e:",E)
                    except:
                        E=0
                        print("no metric yet")
                    # normalize loss by number of batches in validation
                    # todo: normalize over batch_size?
                    loss_list.append([loss.cpu().numpy(), v_loss.cpu().numpy() / i])
                if E>best:
                    best = E
                if E > o_best[0]:
                    o_best[0] = E
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': loss_list,
                        'optimizer_params': optimizer_cfg,
                        'training_time': time.time() - t1
                    }, save_path)
                # set network back to training mode
                net.train()
        return best


    study = optuna.create_study(direction="maximize")
    study.optimize(train, n_trials=30)


if __name__ == '__main__':
    myapp()