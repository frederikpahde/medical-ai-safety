import torch
import numpy as np
import h5py
import os
from tqdm import tqdm
from crp.concepts import ChannelConcept

def precompute_activations_and_relevances(model, dataset, attribution, composite, split, layer_names, config):
    for class_idx in range(len(dataset.classes)):
        run_class_specific_preprocessing(model, dataset, class_idx, attribution, composite, split, layer_names, config)

def run_class_specific_preprocessing(model, dataset, class_idx, attribution, composite, split, layer_names, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
        }
    dataset_split = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])

    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    results_dir = config["dir_precomputed_data"]
    batch_size = config["batch_size"]
    path = f"{results_dir}/global_relevances_and_activations/{dataset_name}/{model_name}"
    str_class_id = 'all' if class_idx is None else class_idx
    os.makedirs(path, exist_ok=True)
    path_h5py = f"{path}/class_{str_class_id}_{split}.hdf5"

    samples = np.array([i for i in range(len(dataset_split)) if ((class_idx is None) or (dataset_split.get_target(i) == class_idx))])
    print(f"Found {len(samples)} samples of class {class_idx}.")

    n_samples = len(samples)
    n_batches = int(np.ceil(n_samples / batch_size))

    cc = ChannelConcept()

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

        condition = [{"y": y} for y in ys]
        attr = attribution(data, condition, composite, record_layer=layer_names)
        output.append(attr.prediction.detach().cpu())

        smpls += [s for s in samples_batch]
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