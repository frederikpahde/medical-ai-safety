import copy
import os
import shutil

import yaml

config_dir = "config_files/training/ptb_xl_attacked"

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
    'dataset_name': 'ptb_xl_attacked',
    'loss': 'binary_cross_entropy',
    'img_size': 224,
    'wandb_api_key': 'YOUR_WANDB_KEY',
    'wandb_project_name': 'ptb-attacked-training',
    'pretrained': False,
    'milestones': "50,75",
    # "attacked_classes": [1],
    # "artifact_type": "defective_lead",
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
    for leads in [
        [1], 
        [2], 
        [3],
        [4]
        ]:
        base_config["lead_ids"] = leads

        for p_artifact in [
            # 0.0,
            0.2,
            0.5
            # 1.0
            ]:
            base_config["p_artifact"] = p_artifact
            lrs = [
                0.001, 
                # 0.005, 
                # 0.0005
                ] #if model_name in ["efficientnet_b0", "efficientnet_b4"] else [0.005, 0.001]
            for lr in lrs:
                base_config['lr'] = lr
                optims = ["sgd"] if "vgg" in model_name else ["adam"] #if "efficientnet" in model_name else ["sgd"]
                for optim_name in optims:
                    base_config['optimizer'] = optim_name

                    for attacked_classes in [
                        [1],
                        [4]
                    ]:
                        base_config['attacked_classes'] = attacked_classes

                        for artifact_type in [
                            "defective_lead",
                            # "amplified_beat"
                        ]:
                            base_config['artifact_type'] = artifact_type
                            config = copy.deepcopy(base_config)
                            str_leads = "-".join([str(l) for l in leads])
                            config_name_base = f"{model_name}_{optim_name}_lr{lr}_attacked{attacked_classes[0]}_leads{str_leads}_p_artifact{p_artifact}_{artifact_type}"

                            if artifact_type == "defective_lead":
                                for seq_length in [
                                    # 0,
                                    100, 
                                    ]:
                                    config["seq_length"] = seq_length
                                    config_name = f"{config_name_base}_seq{seq_length}"
                                    store_local(config, config_name)
                            elif artifact_type == "amplified_beat":
                                for peak_ids in [
                                    # [2],
                                    [1,2,3]
                                    ]:
                                    config["peak_ids"] = peak_ids
                                    for amplify_factor in [
                                        # 2,
                                        3
                                        ]:
                                        config["amplify_factor"] = amplify_factor
                                        str_peak_ids = "-".join([str(p) for p in peak_ids])
                                        config_name = f"{config_name_base}_peak{str_peak_ids}_amp{amplify_factor}"
                                        store_local(config, config_name)
