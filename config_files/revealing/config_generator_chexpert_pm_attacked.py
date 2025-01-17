import copy
import os
import shutil
import yaml
import json

from utils.layer_names import LAYER_NAMES_BY_MODEL

config_dir = "config_files/revealing/chexpert_attacked"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

with open("config_files/chosen_model_params.json", "r") as stream:
    chosen_model_params = json.load(stream)["chexpert_attacked"]

base_config = {
    'dataset_name': 'chexpert_pm_attacked',
    'img_size': 224,
    'method': 'AClarc',
    'wandb_api_key': 'YOUR_WANDB_KEY',
    'wandb_project_name': 'chexpert-pm-attacked-data-annotation',
    'artifact_type': "white_color",
    'attacked_classes': [1]
}


def store_local(config, config_name):
    config['device'] = local_config['local_device']
    model_name = config['model_name']
    optim, lr, p_artifact, alpha = chosen_model_params[model_name]
    config['ckpt_path'] = f"PATH"
    config['batch_size'] = local_config['local_batch_size']
    config['p_artifact'] = p_artifact
    config['alpha'] = alpha
    config['data_paths'] = [local_config['chexpert_dir']]
    config['checkpoint_dir_corrected'] = local_config['checkpoint_dir_corrected']
    config['dir_precomputed_data'] = local_config['dir_precomputed_data']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name, layer_names in [
    ('resnet50d', LAYER_NAMES_BY_MODEL["resnet"][1:]),
    ('vgg16', LAYER_NAMES_BY_MODEL["vgg16"][1:]),
    # ('vit_b_16_torchvision', LAYER_NAMES_BY_MODEL["vit"]),
]:
    
    base_config['model_name'] = model_name
            
    for layer_name in layer_names:
        base_config['layer_name'] = layer_name

        for binary_target in [
            "Cardiomegaly"
        ]:
            base_config["binary_target"] = binary_target
            str_binary = "multilabel" if binary_target is None else f"binaryTarget-{binary_target.replace(' ', '_')}"
            config_name = f"{model_name}_{str_binary}_pm_{layer_name}"
            store_local(base_config, config_name)