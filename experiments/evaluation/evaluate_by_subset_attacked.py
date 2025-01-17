import os
from argparse import ArgumentParser

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import get_dataset, get_dataset_kwargs, load_dataset
from experiments.evaluation.compute_metrics import compute_metrics, compute_model_scores
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs
from utils.helper import load_config

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', 
                        default="config_files/bias_mitigation_controlled/hyper_kvasir_attacked/local/resnet50d_RRClarc_lamb1000000_adam_lr0.001_identity_2.yaml")
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    config = load_config(args.config_file)

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    config['config_file'] = args.config_file

    evaluate_by_subset_attacked(config)


def evaluate_by_subset_attacked(config):
    """ Run evaluations for each data split (train/val/test) on 3 variants of datasets:
            1. Same as training (one attacked class)
            2. Attacked (artifact in all classes)
            3. Clean (no artifacts)

    Args:
        config (dict): config for model correction run
    """
    config_name = config["config_name"]

    print(f"Evaluating {config_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config['dataset_name']
    model_name = config['model_name']

    data_paths = config['data_paths']
    batch_size = config['batch_size']
    img_size = config.get("img_size", 224)
    artifact_type = config.get("artifact_type", None)
    binary_target = config.get("binary_target", None)
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)

    dataset = load_dataset(config, normalize_data=True)

    n_classes = len(dataset.classes)
    ckpt_path = f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path, device=device)
    model = prepare_model_for_evaluation(model, dataset, device, config)

    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }

    dataset_clean = get_dataset(dataset_name)(data_paths=data_paths,
                                              normalize_data=True,
                                              attacked_classes=[],
                                              binary_target=binary_target,
                                              image_size=img_size,
                                              **artifact_kwargs, **dataset_specific_kwargs)

    if "imagenet" in dataset_name:
        all_classes = list(dataset.label_map.keys())
        if config.get("subset_correction", False):
            sets['test'] = sets['test'][::10]
            sets['val'] = sets['val'][::10]
    else:
        all_classes = dataset.classes if "isic" in config["dataset_name"] else range(len(dataset.classes))

    dataset_attacked = get_dataset(dataset_name)(data_paths=data_paths,
                                                 normalize_data=True,
                                                 p_artifact=1.0,
                                                 image_size=img_size,
                                                 artifact_type=artifact_type,
                                                 binary_target=binary_target,
                                                 attacked_classes=all_classes,
                                                 **artifact_kwargs, **dataset_specific_kwargs)
    for split in [
        'test', 
        'val'
        ]:
        split_set = sets[split]

        dataset_ch_split = dataset.get_subset_by_idxs(split_set)
        dataset_clean_split = dataset_clean.get_subset_by_idxs(split_set)
        dataset_attacked_split = dataset_attacked.get_subset_by_idxs(split_set)

        dl_clean = DataLoader(dataset_clean_split, batch_size=batch_size, shuffle=False)
        model_outs_clean, y_true_clean = compute_model_scores(model, dl_clean, device)

        dl = DataLoader(dataset_ch_split, batch_size=batch_size, shuffle=False)
        model_outs, y_true = compute_model_scores(model, dl, device)

        dl_attacked = DataLoader(dataset_attacked_split, batch_size=batch_size, shuffle=False)
        model_outs_attacked, y_true_attacked = compute_model_scores(model, dl_attacked, device)

        

        classes = dataset.classes #None

        metrics = compute_metrics(model_outs, y_true, classes, prefix=f"{split}_", suffix=f"_ch")

        metrics_attacked = compute_metrics(model_outs_attacked, y_true_attacked, classes,
                                           prefix=f"{split}_", suffix=f"_attacked")
        metrics_clean = compute_metrics(model_outs_clean, y_true_clean, classes, prefix=f"{split}_",
                                        suffix=f"_clean")

        if config.get('wandb_api_key', None):
            wandb.log({**metrics, **metrics_attacked, **metrics_clean})


if __name__ == "__main__":
    main()
