import copy
import os
import shutil
import yaml
import json

from utils.layer_names import LAYER_NAMES_BY_MODEL

config_dir = "config_files/bias_mitigation_controlled/isic_attacked"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

with open("config_files/chosen_model_params.json", "r") as stream:
    chosen_model_params = json.load(stream)["isic_attacked"]


base_config = {
    'num_epochs': 10,
    'dataset_name': 'isic_attacked',
    'loss': 'cross_entropy',
    'wandb_api_key': 'YOUR_WANDB_KEY',
    'img_size': 224,
    'wandb_project_name': 'r2r-isic-attacked-mitigation',
    'plot_alignment': False,
    'artifact': 'artificial',
    'artifact_type': "microscope",
    'attacked_classes': ['MEL'],
    'p_artifact': .2,
}

path_local_checkpoints = "/media/pahde/Data/r2r-extended/models/isic_attacked_microscope_v1"

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    model_name = config['model_name']
    optim, lr = chosen_model_params[model_name]
    config['ckpt_path'] = f"{path_local_checkpoints}/{model_name}_p_artifact0.2_p_backdoor0_{optim}_lr{lr}_last.pth"
    config['batch_size'] = local_config['local_batch_size']
    config['data_paths'] = [local_config['isic2019_dir']]
    config['checkpoint_dir_corrected'] = local_config['checkpoint_dir_corrected']
    config['dir_precomputed_data'] = local_config['dir_precomputed_data']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name, layer_names in [
    ('vgg16', ["features.8"] + LAYER_NAMES_BY_MODEL["vgg16"][-5:]),
    ('resnet50d', LAYER_NAMES_BY_MODEL["resnet"][-3:]),
    # ('resnet101d', LAYER_NAMES_BY_MODEL["resnet"]),
    # ('vit_b_16_torchvision', LAYER_NAMES_BY_MODEL["vit"]),
]:
    
    base_config['model_name'] = model_name
    optim_name, lr = chosen_model_params[model_name]
    
    base_config['optimizer'] = optim_name
    base_config['lr'] = lr
        
    for layer_name in layer_names:
        base_config['layer_name'] = layer_name

        ## Vanilla
        config_vanilla = copy.deepcopy(base_config)
        method = 'Vanilla'
        config_vanilla['method'] = method
        config_vanilla['lamb'] = 0.0
        config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_{layer_name}"
        store_local(config_vanilla, config_name)

        config_vanilla = copy.deepcopy(base_config)
        config_vanilla['method'] = method
        config_vanilla['lamb'] = 0.0
        config_vanilla['num_epochs'] = 0
        config_name = f"{model_name}_{method}-0epochs_{optim_name}_lr{lr}_{layer_name}"
        store_local(config_vanilla, config_name)

        ## ClArC
        method = "RRClarc"
        config_clarc = copy.deepcopy(base_config)
        config_clarc["direction_mode"] = "signal"
        config_clarc["cav_mode"] = "cavs_max"
        config_clarc["cav_scope"] = [0]
        config_clarc["method"] = method
        config_clarc["criterion"] = "all_logits_random"
        lambs = [
                1e1, 5 * 1e1,
                # 1e2, 5 * 1e2,
                # 1e3, 5 * 1e3,
                # 1e4, 5 * 1e4,
                # 1e5, 5 * 1e5,
                # 1e6, 5 * 1e6,
                # 1e7, 5 * 1e7,
                # 1e8, 5 * 1e8,
                # 1e9, 5 * 1e9,
                # 1e10, 5 * 1e10,
                # 1e11, 5 * 1e11,
                # 1e12, 5 * 1e12,
            ] 

        for lamb in lambs:
            config_clarc["lamb"] = lamb
            config_name = f"{model_name}_{method}_lamb{lamb:.0f}_{optim_name}_lr{lr}_{layer_name}"
            store_local(config_clarc, config_name)

        ## P-ClArC
        method = "PClarc"
        config_clarc = copy.deepcopy(base_config)
        config_clarc["direction_mode"] = "signal"
        config_clarc["cav_mode"] = "cavs_max"
        config_clarc["cav_scope"] = [0]
        config_clarc["method"] = method
        config_clarc["num_epochs"] = 0
        config_clarc["criterion"] = "all_logits_random"
        for lamb in [
            1
        ]:
            config_clarc["lamb"] = lamb
            config_name = f"{model_name}_{method}_lamb{lamb:.0f}_{optim_name}_lr{lr}_{layer_name}"
            store_local(config_clarc, config_name)

         ## P-ClArC
        method = "ReactivePClarc"
        config_clarc = copy.deepcopy(base_config)
        config_clarc["direction_mode"] = "signal"
        config_clarc["cav_mode"] = "cavs_max"
        config_clarc["cav_scope"] = [0]
        config_clarc["method"] = method
        config_clarc["num_epochs"] = 0
        config_clarc["criterion"] = "all_logits_random"
        for lamb in [
            1
        ]:
            config_clarc["lamb"] = lamb
            config_name = f"{model_name}_{method}_lamb{lamb:.0f}_{optim_name}_lr{lr}_{layer_name}"
            store_local(config_clarc, config_name)

        ## RRR
        method = "RRR_ExpMax"
        config_rrr = copy.deepcopy(base_config)
        config_rrr["method"] = method
        lambs = [
                1e1, 5 * 1e1,
                # 1e2, 5 * 1e2,
                # 1e3, 5 * 1e3,
                # 1e4, 5 * 1e4,
                # 1e5, 5 * 1e5,
                # 1e6, 5 * 1e6,
                # 1e7, 5 * 1e7,
                # 1e8, 5 * 1e8,
                # 1e9, 5 * 1e9,
            ] 
        
        for lamb in lambs:
            for source_maks in [
                "gt", 
                # "bin", 
                # "hm"
                ]:
                config_rrr["lamb"] = lamb
                config_rrr["source_maks"] = source_maks
                config_name = f"{model_name}_{method}_{source_maks}_lamb{lamb:.0f}_{optim_name}_lr{lr}_{layer_name}"
                store_local(config_rrr, config_name)