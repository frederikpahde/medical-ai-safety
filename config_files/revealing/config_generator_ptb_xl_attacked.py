import copy
import os
import shutil
import yaml
import json

from utils.layer_names import LAYER_NAMES_BY_MODEL

config_dir = "config_files/revealing/ptb_xl_attacked"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

with open("config_files/chosen_model_params.json", "r") as stream:
    chosen_model_params = json.load(stream)["ptb_xl_attacked"]

base_config = {
    'method': 'AClarc',
    'device': 'cuda',
    'dataset_name': 'ptb_xl_attacked',
    'img_size': 224,
    'wandb_api_key': 'YOUR_WANDB_KEY',
    'wandb_project_name': 'ptb-xl-attacked-data-annotation',
    'cav_mode': 'cavs_max',
    "attacked_classes": [1],
    "artifact_type": "defective_lead",
}

def store_local(config, config_name):
    model_name = config['model_name']
    optim, lr, seq_length, p_artifact, lead_ids = chosen_model_params[model_name]
    config['ckpt_path'] = f"PATH"
    config['seq_length'] = seq_length
    config['p_artifact'] = p_artifact
    config['lead_ids'] = lead_ids
    config['batch_size'] = local_config['local_batch_size']
    config['data_paths'] = [local_config['ptb_xl_dir']]
    config['checkpoint_dir_corrected'] = local_config['checkpoint_dir_corrected']
    config['dir_precomputed_data'] = local_config['dir_precomputed_data']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name, layer_names in [
    ('xresnet1d50', LAYER_NAMES_BY_MODEL["xresnet1d50"]),
]:
    
    base_config['model_name'] = model_name
            
    for layer_name in layer_names:
        base_config['layer_name'] = layer_name

        config_name = f"{model_name}_{layer_name}"
        store_local(base_config, config_name)