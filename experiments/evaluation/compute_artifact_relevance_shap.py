import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
import wandb
from tqdm import tqdm

from datasets import load_dataset
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader
from utils.helper import load_config
from captum.attr import ShapleyValueSampling
from fast_slic import Slic

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/bias_mitigation_controlled/hyper_kvasir_attacked/local/vit_b_16_torchvision_RRClarc_lamb100_sgd_lr0.001_inspection_layer.yaml")

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    config = load_config(args.config_file)

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    compute_artifact_relevance(config)


def compute_artifact_relevance(config):
    """
    Computes average relevance in artifactual regions for train/val/test splits.

    Args:
        config (dict): experiment config
    """
    print(f"In computing artifact Rel.")
    model_name = config['model_name']
    config_name = config['config_name']

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config.get('device', default_device)

    dataset = load_dataset(config, normalize_data=True, hm=True)

    ckpt_path = f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"
    model = get_fn_model_loader(model_name=model_name)(n_class=dataset.classes.__len__(), ckpt_path=ckpt_path, device=device)
    model = prepare_model_for_evaluation(model, dataset, device, config)

    artifact_labels = list(dataset.sample_ids_by_artifact.keys())
    artifact_sample_ids = list(dataset.sample_ids_by_artifact.values())
    scores = []
    sems = []

    train_set = dataset.idxs_train
    test_set = dataset.idxs_test
    val_set = dataset.idxs_val

    splits = {
        "val": val_set,
        "test": test_set,
        "train": train_set
    }

    gaussian = torchvision.transforms.GaussianBlur(kernel_size=41, sigma=8.0)
    svs = ShapleyValueSampling(model)
    slic = Slic(num_components=100, compactness=10)

    for split in ['train', 'val', 'test']:

        split_set = splits[split]
        sample_sets_split = [[y for y in x if y in split_set] for x in artifact_sample_ids]

        for k, samples in enumerate(sample_sets_split):

            dataset = load_dataset(config, normalize_data=True, hm=True)

            n_samples = len(samples)
            n_batches = int(np.ceil(n_samples / config['batch_size']))
            score = []
            for i in tqdm(range(n_batches)):
                samples_batch = samples[i * config['batch_size']:(i + 1) * config['batch_size']]
                data = torch.stack([dataset[j][0] for j in samples_batch], dim=0)
                targets = torch.stack([dataset[j][1] for j in samples_batch], dim=0)
                
                # Compute super-pixel assignment with FastSlic
                assignments_batch = None
                for j in range(len(data)):
                    sample_np = dataset.reverse_normalization(data[j]).numpy().transpose(1,2,0).astype("uint8")
                    assignment = torch.tensor(slic.iterate(sample_np.copy(order='C'))).to(device).unsqueeze(0)
                    assignments_batch = assignment if assignments_batch is None else torch.cat([assignments_batch, assignment])

                attr_batch = svs.attribute(data.to(device), target=targets.to(device), 
                                n_samples=150, feature_mask=assignments_batch.unsqueeze(1).to(device), 
                                show_progress=True, baselines=0)

                hm = attr_batch.sum(1).detach()

                # load mask as third entry from data sample
                mask = torch.stack([dataset[j][2] for j in samples_batch], dim=0).to(device)
                mask = 1.0 * (mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None] > 0.1)
                mask = gaussian(mask.clamp(min=0)) ** 1.0
                mask = 1.0 * (mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None] > 0.3)

                inside = (hm * mask).abs().sum((1, 2)) / (hm.abs().sum((1, 2)) + 1e-10)

                score.extend(list(inside.detach().cpu()))

            scores.append(np.mean(score))
            sems.append(np.std(score) / np.sqrt(len(score)))
            
            if config.get('wandb_api_key', None):
                print(f"Logging artifact rel ({artifact_labels[k].lower()}) for split {split}: {scores[-1]:.3f}")
                wandb.log({f"{split}_artifact_rel_{artifact_labels[k].lower()}": scores[-1]})
                wandb.log({f"{split}_artifact_rel_{artifact_labels[k].lower()}_sem": sems[-1]})


if __name__ == "__main__":
    main()
