import os
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from datasets import load_dataset
from experiments.evaluation.compute_metrics import compute_metrics, compute_model_scores
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader
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


    evaluate_by_subset(config)

def evaluate_by_subset(config):
    """Run evaluations for all data splits and sets of artifacts

    Args:
        config (dict): model correction run config
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config['dataset_name']
    model_name = config['model_name']
    config_name = config['config_name']
    limit_train_batches = config.get("limit_train_batches", None)

    dataset = load_dataset(config, normalize_data=True)

    n_classes = len(dataset.classes)

    ckpt_path = config["ckpt_path"]
    if config.get("num_epochs",0) == 0 and dataset_name == "imagenet":
        ckpt_path = None

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path, device=device)
    model = prepare_model_for_evaluation(model, dataset, device, config)

    sets = {
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
    }

    labels_set = list(dataset.sample_ids_by_artifact.keys())
    sample_sets = list(dataset.sample_ids_by_artifact.values())

    all_idxs = np.arange(len(dataset))
    sample_sets.append(all_idxs)
    labels_set.append("all")
    for k, v in dataset.sample_ids_by_artifact.items():
        clean_idxs = np.setdiff1d(all_idxs, v)
    sample_sets.append(clean_idxs)
    labels_set.append("clean")

    for split in ['test', 'val']:
        split_set = sets[split]
        sample_sets_split = [[y for y in x if y in split_set] for x in sample_sets]

        model_outs_all = []
        ys_all = []
        print(f"size of sample sets ({split})", [len(x) for x in sample_sets_split])

        for k, samples in enumerate(sample_sets_split):
            if len(samples) > 0:
                samples = np.array(samples)
                dataset_subset = dataset.get_subset_by_idxs(samples)
                dl_subset = DataLoader(dataset_subset, batch_size=config['batch_size'], shuffle=False)
                model_outs, y_true = compute_model_scores(model, dl_subset, device, limit_train_batches)
                classes = None if "imagenet" in dataset_name else dataset.classes
                metrics = compute_metrics(model_outs, y_true, classes,
                                        prefix=f"{split}_",
                                        suffix=f"_{labels_set[k].lower()}")
                model_outs_all.append(model_outs)
                ys_all.append(y_true)
                if config['wandb_api_key']:
                    print('logging', metrics)
                    wandb.log(metrics)

        model_outs_all = torch.cat(model_outs_all)
        ys_all = torch.cat(ys_all)
        classes = None if "imagenet" in dataset_name else dataset.classes
        metrics_all = compute_metrics(model_outs_all, ys_all, classes, prefix=f"{split}_")
        if config.get('wandb_api_key', None):
            print('logging', metrics_all)
            wandb.log(metrics_all)

if __name__ == "__main__":
    main()
