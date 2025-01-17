import copy
import os
import shutil

import yaml

config_dir = "config_files/training/chexpert_attacked"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

base_config = {
    'num_epochs': 300,
    'eval_every_n_epochs': 10,
    'store_every_n_epochs': 500,
    'dataset_name': 'chexpert_pm_attacked',
    'img_size': 224,
    'wandb_api_key': 'YOUR_WANDB_KEY',
    'wandb_project_name': 'chexpert-attacked-training',
    'pretrained': True,
    'milestones': "150,250",
    'artifact_type': "white_color",
    'attacked_classes': [1]
    # "percentage_batches": .05
}

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['chexpert_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name in [
    'vgg16',
    'resnet50d',
    'vit_b_16_torchvision',
    # 'resnet101d',
]:
    base_config['model_name'] = model_name
    lrs = [ 
        0.0005,
        0.001, 
        0.005,
        ] #if model_name in ["efficientnet_b0", "efficientnet_b4"] else [0.005, 0.001]
    for lr in lrs:
        base_config['lr'] = lr

        if "resnet" in model_name:
            optims = ["adam"]
        elif "vgg" in model_name:
            optims = ["sgd"]
        else:
            optims = ["adam", "sgd"]
        for optim_name in optims:
            for binary_target in [
                # "No Finding",
                "Cardiomegaly",
                # None
            ]:
                for p_artifact in [.05, .1]:
                    for alpha in [.2]:
                        base_config['p_artifact'] = p_artifact
                        base_config['alpha'] = alpha
                        base_config['optimizer'] = optim_name
                        base_config['epochs'] = 50 if binary_target is None else 150
                        base_config['milestones'] = "30,40" if binary_target is None else "80,120"
                        base_config['binary_target'] = binary_target
                        base_config["loss"] = "binary_cross_entropy" if binary_target is None else "cross_entropy"
                        config_name = f"{model_name}_{optim_name}_lr{lr}_p_artifact{p_artifact}_alpha{alpha}"
                        config_name += f"_binaryTarget-{binary_target.replace(' ', '_')}_pm" if binary_target else "_pm"
                        store_local(base_config, config_name)
