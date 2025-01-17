import copy
import os
import shutil

import yaml

config_dir = "config_files/training/isic_attacked"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)


base_config = {
    'num_epochs': 300,
    'device': 'cuda',
    'eval_every_n_epochs': 5,
    'store_every_n_epochs': 100,
    'dataset_name': 'isic_attacked',
    'loss': 'cross_entropy',
    'img_size': 224,
    'wandb_api_key': 'YOUR_WANDB_KEY',
    'wandb_project_name': 'isic-attacked-microscope',
    'attacked_classes': ['MEL'],
    'pretrained': True,
    'artifact': 'artificial',
    'artifact_type': "microscope",
    'milestones': "150,250"
}

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['isic2019_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name in [
    'vgg16',
    'resnet50d',
]:
    base_config['model_name'] = model_name
    for p_artifact in [
        # 0.01,
        .2,
        ]:
        for p_backdoor in [0, 
                        #    0.001
                           ]:
            base_config["p_artifact"] = p_artifact
            base_config["p_backdoor"] = p_backdoor
            lrs = [0.0005, 0.001, 0.005]# if model_name in ["efficientnet_b0"] else [0.001, 0.005]
            for lr in lrs:
                base_config['lr'] = lr
                optims = ["adam", "sgd"]# if "efficientnet" in model_name else ["sgd"]
                for optim_name in optims:
                    base_config['optimizer'] = optim_name
                    config_name = f"{model_name}_p_artifact{p_artifact}_p_backdoor{p_backdoor}_{optim_name}_lr{lr}"
                    store_local(base_config, config_name)
