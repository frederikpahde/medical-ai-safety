import logging
import os
from argparse import ArgumentParser

import torch
import tqdm
import wandb
from torch.utils.data import DataLoader

from datasets import load_dataset
from experiments.evaluation.compute_metrics import aggregate_tcav_metrics, compute_tcav_metrics_batch
from models import get_fn_model_loader
from utils.cav_utils import get_cav_from_model
from utils.helper import load_config

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/bias_mitigation_controlled/hyper_kvasir_attacked/local/resnet50d_RRClarc_lamb1000000_adam_lr0.001_identity_2.yaml")
    parser.add_argument('--before_correction', action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    config = load_config(args.config_file)

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    measure_quality_cav(config, args.before_correction)

def get_activation(module, input_, output_):
            global activations
            activations = output_
            return output_.clone()

def measure_quality_cav(config, before_correction):
    """ Computes TCAV scores
    Args:
        config (dict): config for model correction run
    """

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    model_name = config['model_name']
    config_name = config['config_name']
    artifact_name = config["artifact"]
    config["device"] = device

    ### compute TCAV on last layer
    if "resnet" in model_name:
        config["layer_name"] = "identity_2"
    elif "vgg" in model_name:
        config["layer_name"] = "features.29"
    elif "vit" in model_name:
        config["layer_name"] = "inspection_layer"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    dataset = load_dataset(config, normalize_data=True)
    n_classes = len(dataset.classes)
    ckpt_path =  config['ckpt_path'] if before_correction else f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path, device=device).to(device)

    # Get CAV
    cav = get_cav_from_model(model, dataset, config, config['artifact'])

    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }
    
    model.eval()

    results = {}
    for split in [
        'train',
        'test',
        'val'
    ]:
        split_set = sets[split]

        artifact_ids_split = [i for i in dataset.sample_ids_by_artifact[artifact_name] if i in split_set]
        dataset_artifact_only = dataset.get_subset_by_idxs(artifact_ids_split)

        dl_art = DataLoader(dataset_artifact_only, batch_size=1, shuffle=False)

        # Register forward hook for layer of interest
        layer = config["layer_name"]
        for n, m in model.named_modules():
            if n.endswith(layer):
                m.register_forward_hook(get_activation)

        # controlled setting
        attacked_class = dataset.get_class_id_by_name(dataset.attacked_classes[0])
        
        TCAV_sens_list = []
        TCAV_pos = 0
        TCAV_neg = 0
        TCAV_pos_mean = 0
        TCAV_neg_mean = 0
        for sample in tqdm.tqdm(dl_art):
            x_att, y = sample
            grad_target = attacked_class if attacked_class is not None else y

            # Compute latent representation
            with torch.enable_grad():
                x_att.requires_grad = True
                x_att = x_att.to(device)
                y_hat = model(x_att)
                yc_hat = y_hat[:, grad_target]

                grad = torch.autograd.grad(outputs=yc_hat,
                                           inputs=activations,
                                           retain_graph=True,
                                           grad_outputs=torch.ones_like(yc_hat))[0]

                grad = grad.detach().cpu()
                model.zero_grad()

                tcav_metrics_batch = compute_tcav_metrics_batch(grad, cav)
                        
                TCAV_pos += tcav_metrics_batch["TCAV_pos"]
                TCAV_neg += tcav_metrics_batch["TCAV_neg"]
                TCAV_pos_mean += tcav_metrics_batch["TCAV_pos_mean"]
                TCAV_neg_mean += tcav_metrics_batch["TCAV_neg_mean"]

                TCAV_sens_list.append(tcav_metrics_batch["TCAV_sensitivity"])

        
        tcav_metrics = aggregate_tcav_metrics(TCAV_pos, TCAV_neg, TCAV_pos_mean, TCAV_neg_mean, TCAV_sens_list)

        metric_extension = "_before_correction" if before_correction else ""
        results[f"{split}_mean_tcav_quotient_{artifact_name}{metric_extension}"] = tcav_metrics['mean_tcav_quotient']
        results[f"{split}_mean_quotient_{artifact_name}_sderr{metric_extension}"] = tcav_metrics['mean_quotient_sderr']

        results[f"{split}_tcav_quotient_{artifact_name}{metric_extension}"] = tcav_metrics['tcav_quotient']
        results[f"{split}_quotient_{artifact_name}_sderr{metric_extension}"] = tcav_metrics['quotient_sderr']

        results[f"{split}_mean_tcav_sensitivity_{artifact_name}{metric_extension}"] = tcav_metrics['mean_tcav_sensitivity']
        results[f"{split}_mean_tcav_sensitivity_{artifact_name}_sem{metric_extension}"] = tcav_metrics['mean_tcav_sensitivity_sem']

        if config.get('wandb_api_key', None):
            wandb.log({**results, **config})


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
