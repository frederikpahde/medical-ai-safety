import torch

def get_optimizer(optim_name, params, lr, ckpt_path=None):
    if optim_name == 'sgd':
        optim = torch.optim.SGD(params, lr, momentum=0.9)
    elif optim_name == 'adam':
        optim = torch.optim.Adam(params, lr, eps=1e-07)
    elif optim_name == 'adamw':
        optim = torch.optim.AdamW(params, lr, eps=1e-07)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")
    
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        if "optimizer_state_dict" in checkpoint.keys():
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
    return optim

def get_loss(loss_name, weights=None):
    losses = {
        'cross_entropy': torch.nn.CrossEntropyLoss(weight=weights),
        'binary_cross_entropy': torch.nn.BCEWithLogitsLoss(weight=weights)
    }
    assert loss_name in losses.keys(), f"Loss '{loss_name}' not supported, select one of the following: {list(losses.keys())}"
    return losses[loss_name]