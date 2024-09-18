import importlib
import torch
from lion_pytorch import Lion

from third_party.decode.models import SigmaMUNet


def load(model_path, optimizer_cfg, network_cfg, device, decode=False):
    if decode:
        net = SigmaMUNet(3)
        print("loading Decode")
    else:
        net_package = importlib.import_module("models.VIT." + network_cfg.name.lower())
        net = net_package.Network(network_cfg.components)
        print("loading network {}".format(network_cfg.name))
    loss = None
    epoch=0
    if optimizer_cfg["name"] == "Lion":
        opt = Lion(net.parameters(), **optimizer_cfg["params"])
    else:
        opt_cls = getattr(torch.optim, optimizer_cfg["name"])
        opt = opt_cls(net.parameters(), **optimizer_cfg["params"])
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