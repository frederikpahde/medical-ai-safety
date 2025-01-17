import copy
import os
import shutil

import yaml

config_dir = "config_files/training/ptb_xl"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

base_config = {
    'num_epochs': 100,
    'eval_every_n_epochs': 5,
    'store_every_n_epochs': 500,
    'dataset_name': 'ptb_xl',
    'loss': 'binary_cross_entropy',
    'img_size': 224,
    'wandb_api_key': 'YOUR_WANDB_KEY',
    'wandb_project_name': 'ptb-training',
    'pretrained': False,
    'milestones': "50,75",
    # 'percentage_batches': .1
}

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['ptb_xl_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name in [
    'xresnet1d50',
]:
    base_config['model_name'] = model_name
    lrs = [0.001, 0.005, 0.0005] #if model_name in ["efficientnet_b0", "efficientnet_b4"] else [0.005, 0.001]
    for lr in lrs:
        base_config['lr'] = lr
        optims = ["sgd"] if "vgg" in model_name else ["adam", "sgd"] #if "efficientnet" in model_name else ["sgd"]
        for optim_name in optims:
            base_config['optimizer'] = optim_name
            config_name = f"{model_name}_{optim_name}_lr{lr}"
            store_local(base_config, config_name)
