import logging
import os
from argparse import ArgumentParser

import h5py
import numpy as np
import torch
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import load_dataset
from models import MODELS_1D, get_canonizer, get_fn_model_loader, TRANSFORMER_MODELS

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', default="config_files/revealing/isic/local/resnet50d_identity_2.yaml")
    parser.add_argument("--class_id", default=8, type=int)
    parser.add_argument("--force_recompute", default=False, type=bool)
    parser.add_argument("--split", default="all", choices=['train', 'val', 'test', 'all'], type=str)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)

    config['config_file'] = args.config_file
    config['split'] = args.split
    config['class_idx'] = args.class_id

    run_collect_relevances_and_activations(config, args.force_recompute)


def run_collect_relevances_and_activations(config, force_recompute):
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    dataset_name = config['dataset_name']
    p_artifact = config.get("p_artifact", None)
    artifact_type = config.get("artifact_type", None)
    split = config["split"]
    model_name = config['model_name']
    ckpt_path = config["ckpt_path"]
    batch_size = config['batch_size']
    results_dir = config.get('dir_precomputed_data', 'results')
    class_idx = config['class_idx']
    artifact_extension = f"_{artifact_type}-{p_artifact}" if p_artifact is not None else ""
    path = f"{results_dir}/global_relevances_and_activations/{dataset_name}{artifact_extension}/{model_name}"
    str_class_id = 'all' if class_idx is None else class_idx
    path_h5py = f"{path}/class_{str_class_id}_{split}.hdf5"

    if (not force_recompute) and (os.path.isfile(path_h5py)):
        print(f"Global activations/relevances are already computed: {path_h5py}")
        return

    
    os.makedirs(path, exist_ok=True)

    dataset = load_dataset(config, normalize_data=True)

    splits = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
        }
    dataset_split = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])
    
    logger.info(f"Using split {split} ({len(dataset_split)} samples)")

    n_classes = len(dataset_split.classes)

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path, device=device)
    model = model.to(device)
    model.eval()

    attribution = CondAttribution(model)
    canonizers = get_canonizer(model_name)

    if "densenet" in model_name:
        canonizers[0].apply(model)
        composite = EpsilonPlusFlat()
    else:
        composite = EpsilonPlusFlat(canonizers)

    cc = ChannelConcept()

    # Find sample IDs with given class_idx
    samples = np.array([i for i in range(len(dataset_split)) if ((class_idx is None) or (dataset_split.get_target(i) == class_idx))])
    logger.info(f"Found {len(samples)} samples of class {class_idx}.")

    n_samples = len(samples)
    n_batches = int(np.ceil(n_samples / batch_size))

    if ("resnet" in model_name) or ("efficientnet" in model_name):
        layer_names = [n for n, m in model.named_modules() if (isinstance(m, torch.nn.Identity) and 
                                                               ("identity" in n) or "last_conv" in n)]
    elif any([m in model_name for m in TRANSFORMER_MODELS]):
        layer_names = ["inspection_layer"]
    else:
        layer_names = [n for n, m in model.named_modules() if isinstance(m, (torch.nn.Identity, torch.nn.Conv2d, torch.nn.ReLU))]

    print(f"Found {len(layer_names)} layers for model '{model_name}' ({layer_names})")
    
    rels = dict(zip(layer_names, [[] for _ in layer_names]))
    rels_max = dict(zip(layer_names, [[] for _ in layer_names]))
    rels_mean = dict(zip(layer_names, [[] for _ in layer_names]))
    cavs_max = dict(zip(layer_names, [[] for _ in layer_names]))
    cavs_mean = dict(zip(layer_names, [[] for _ in layer_names]))
    smpls = []
    output = []
    hms = []

    for i in tqdm(range(n_batches)):
        samples_batch = samples[i * batch_size:(i + 1) * batch_size]
        data = torch.stack([dataset_split[j][0] for j in samples_batch], dim=0).to(device).requires_grad_()
        ys = torch.stack([dataset_split[j][1] for j in samples_batch], dim=0).to(device)
        # out = model(data).detach().cpu()
        # condition = [{"y": c_id} for c_id in out.argmax(1)]
        condition = [{"y": y} for y in ys]
        # attr = attribution(data, condition, composite, record_layer=layer_names, init_rel=1)
        attr = attribution(data, condition, composite, record_layer=layer_names)
        output.append(attr.prediction.detach().cpu())

        

        smpls += [s for s in samples_batch]
        if any([n in model_name for n in MODELS_1D]):
            lnames = [l for l in layer_names if l in attr.relevances.keys()]
            acts_max = [attr.activations[layer] for layer in lnames]
            acts_mean = [attr.activations[layer] for layer in lnames]
            rels_batch = [cc.attribute(attr.relevances[layer], abs_norm=True) for layer in lnames]
            rels_max_batch = [attr.relevances[layer] for layer in lnames]
            rels_mean_batch = [attr.relevances[layer] for layer in lnames]
        else:
            lnames = [lname for lname, acts in attr.activations.items() if acts.dim() == 4]
            rels_batch = [cc.attribute(attr.relevances[layer], abs_norm=True) for layer in lnames]
            rels_max_batch = [attr.relevances[layer].flatten(start_dim=2).max(2)[0] for layer in lnames]
            rels_mean_batch = [attr.relevances[layer].mean((2, 3)) for layer in lnames]
            acts_max = [attr.activations[layer].flatten(start_dim=2).max(2)[0] for layer in lnames]
            acts_mean = [attr.activations[layer].mean((2, 3)) for layer in lnames]

        for l, r, rmax, rmean, amax, amean in zip(lnames, rels_batch, rels_max_batch, rels_mean_batch, acts_max, acts_mean):
            rels[l] += r.detach().cpu()
            rels_max[l] += rmax.detach().cpu().clamp(min=0)
            rels_mean[l] += rmean.detach().cpu().clamp(min=0)
            cavs_max[l] += amax.detach().cpu()
            cavs_mean[l] += amean.detach().cpu()

            if l == "input_identity":
                hms += attr.heatmap.detach().cpu()

    
    print("saving as", f"{path}/class_{class_idx}_{split}.pth")

    
    print(f"writing to {path_h5py}")
    f = h5py.File(path_h5py, "w")
    for layer in layer_names:
        layer_group = f.create_group(layer)
        data_rel = torch.tensor([]) if len(rels[layer]) == 0 else torch.stack(rels[layer])
        data_rel_mean = torch.tensor([]) if len(rels_mean[layer]) == 0 else torch.stack(rels_mean[layer])
        data_rel_max = torch.tensor([]) if len(rels_max[layer]) == 0 else torch.stack(rels_max[layer])
        data_act_mean = torch.tensor([]) if len(cavs_mean[layer]) == 0 else torch.stack(cavs_mean[layer])
        data_act_max = torch.tensor([]) if len(cavs_max[layer]) == 0 else torch.stack(cavs_max[layer])
        data_hms = torch.tensor([]) if len(hms) == 0 else torch.stack(hms)
        layer_group.create_dataset("rel", data=data_rel)
        layer_group.create_dataset("rels_max", data=data_rel_max)
        layer_group.create_dataset("rels_mean", data=data_rel_mean)
        layer_group.create_dataset("cavs_max", data=data_act_max)
        layer_group.create_dataset("cavs_mean", data=data_act_mean)
        
        if layer == "input_identity":
            layer_group.create_dataset("hm", data=data_hms)
    f.close()
    data_outputs = torch.tensor([]) if len(output) == 0 else torch.cat(output)
    torch.save({"samples": smpls,
                "output": data_outputs},
               f"{path}/class_{str_class_id}_{split}_meta.pth")

if __name__ == "__main__":
    main()
