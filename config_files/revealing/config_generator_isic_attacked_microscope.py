import copy
import os
import shutil
import yaml
import json
from utils.layer_names import LAYER_NAMES_BY_MODEL

config_dir = "config_files/revealing/isic_attacked_microscope"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)


base_config = {
    'method': 'AClarc',
    'dataset_name': 'isic_attacked',
    'img_size': 224,
    'cav_mode': 'cavs_max',
    'artifact': 'artificial',
    'artifact_type': "microscope",
    'attacked_classes': ['MEL'],
    'p_artifact': .2,
    'wandb_api_key': 'YOUR_WANDB_KEY',
    'wandb_project_name': 'isic-attacked-microscope-data-annotation',
}

with open("config_files/chosen_model_params.json", "r") as stream:
    chosen_model_params = json.load(stream)["isic_attacked"]


def store_local(config, config_name):
    config['device'] = local_config['local_device']
    model_name = config['model_name']
    # optim, lr = configs_by_model_name[model_name]
    optim, lr = chosen_model_params[model_name]
    config['ckpt_path'] = f"PATH"
    config['batch_size'] = local_config['local_batch_size']
    config['data_paths'] = [local_config['isic2019_dir']]
    config['checkpoint_dir_corrected'] = local_config['checkpoint_dir_corrected']
    config['dir_precomputed_data'] = local_config['dir_precomputed_data']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name, layer_names in [
    ('vgg16', LAYER_NAMES_BY_MODEL["vgg16"]),
    ('resnet50d', LAYER_NAMES_BY_MODEL["resnet"]),
    # ('resnet101d', LAYER_NAMES_BY_MODEL["resnet"]),
    # ('densenet121', LAYER_NAMES_BY_MODEL["densenet121"]),
    # ('vit_b_16_torchvision', LAYER_NAMES_BY_MODEL["vit"]),
]:
    
    base_config['model_name'] = model_name
            
    for layer_name in layer_names:
        base_config['layer_name'] = layer_name

        config_name = f"{model_name}_{layer_name}"
        store_local(base_config, config_name)