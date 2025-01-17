import os
from model_correction import get_correction_method
import gc
import copy
import torch
import numpy as np

def get_cav_from_model(model, dataset, config, artifact, mode=None, store_cav=False):
    method = "AClarc"
    kwargs_correction = {}
    config["cav_mode"] = "cavs_max" if "cav_mode" not in config else config["cav_mode"]
    if mode is not None:
        config["cav_mode"] = mode
    config["cav_scope"] = None if "cav_scope" not in config else config["cav_scope"]
    # if "isic" in config["dataset_name"] and (config["cav_scope"] is not None):
    #     config["cav_scope"] = [dataset.classes.index(cl) for cl in config["cav_scope"]]
    config["direction_mode"] = "svm" if "direction_mode" not in config else config["direction_mode"]
    config["lamb"] = None if "lamb" not in config else config["lamb"]
    kwargs_correction['classes'] = dataset.classes if config["dataset_name"] == "isic" else range(len(dataset.classes))
    kwargs_correction['artifact_sample_ids'] = dataset.sample_ids_by_artifact[artifact]
    kwargs_correction['sample_ids'] = np.array([i for i in dataset.idxs_train])  # [i for i in dataset.idxs_val]
    kwargs_correction['mode'] = config["cav_mode"]
    correction_method = get_correction_method(method)
    model_corrected = correction_method(copy.deepcopy(model), config, **kwargs_correction)
    w = model_corrected.cav.clone().detach().cpu().reshape(-1)#.numpy()
    del model_corrected
    torch.cuda.empty_cache(); gc.collect

    # store cav
    if store_cav:
        p_artifact = config.get("p_artifact", None)
        artifact_type = config.get("artifact_type", None)
        artifact_extension = f"_{artifact_type}-{p_artifact}" if p_artifact is not None else ""
        cav_mode = config["cav_mode"].replace("cavs", "cav")
        path = f"{config['dir_precomputed_data']}/cavs/{config['dataset_name']}{artifact_extension}/{config['model_name']}/{cav_mode}_{artifact}_{config['layer_name']}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Storing CAV to {path}")
        torch.save(w, path)

    return w
