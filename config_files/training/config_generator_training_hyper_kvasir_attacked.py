import copy
import os
import shutil

import yaml

config_dir = "config_files/training/hyper_kvasir_attacked"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

base_config = {
    'num_epochs': 300,
    'eval_every_n_epochs': 30,
    'store_every_n_epochs': 500,
    'dataset_name': 'hyper_kvasir_attacked',
    'loss': 'cross_entropy',
    'img_size': 224,
    'wandb_api_key': 'YOUR_WANDB_KEY',
    'wandb_project_name': 'hyper-kvasir-attacked-training',
    'pretrained': True,
    'milestones': "150,250",
    # "p_backdoor": .25,
    "attacked_classes": [1],
    "artifact_type": "ch_time",
    'time_format': "datetime"
}

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['hyper_kvasir_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def store_cluster(config, config_name):
    batch_size = 128
    if "former" in config_name:
        batch_size = 64 
    elif "efficientnet_v2" in config_name:
        batch_size = 64
    config['device'] = "cuda"
    config['batch_size'] = batch_size
    config['model_savedir'] = "/mnt/output"
    config['data_paths'] = ["/mnt/dataset_hyper_kvasir"]

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name in [
    'vgg16',
    'vit_b_16_torchvision',
    'resnet50d',
]:
    base_config['model_name'] = model_name
    lrs = [ 
        0.0005,
        0.001, 
        0.005,
        ] #if model_name in ["efficientnet_b0", "efficientnet_b4"] else [0.005, 0.001]
    for p_artifact in [
        .1,
        # .2,
        # .5
    ]:
        base_config['p_artifact'] = p_artifact
        for lr in lrs:
            base_config['lr'] = lr
            optims = ["adam"] if "resnet" in model_name else ["adam", "sgd"]
            # optims = [
            #     "adam", 
            #     "sgd"
            #     ] #if "efficientnet" in model_name else ["sgd"]
            for optim_name in optims:
                # for pool in ["max", "avg"]:

                #     base_config['pool'] = pool
                base_config['optimizer'] = optim_name
                config_name = f"{model_name}_{optim_name}_lr{lr}_p_artifact{p_artifact}"
                store_local(base_config, config_name)
                store_cluster(base_config, config_name)
