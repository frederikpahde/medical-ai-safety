import yaml
import os
from crp.helper import get_layer_names
import torch.nn as nn

def none_or_int(value):
  if value.lower() == "none":
    return None
  return int(value)

def none_or_str(value):
  if value.lower() == "none":
    return None
  return value

def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["config_name"] = os.path.basename(config_path)[:-5]
            config["wandb_id"] = config["config_name"] 
        except yaml.YAMLError as exc:
            print(exc)
            config = {}
    return config


def get_layer_names_with_identites(model):
    BLACKLIST = ["downsample",
                 "conv1",
                 "conv2",
                 "conv3"]
    layer_names_conv = get_layer_names(model, [nn.Conv2d, nn.ReLU])
    layer_names_conv = [l for l in layer_names_conv if not any([b in l for b in BLACKLIST])]
    layer_names_identity = [l for l in get_layer_names(model, [nn.Identity])
                            if any([s in l for s in ["last_conv", "inspection_layer", "identity",
                                                     "features.4.3", # identity_0
        "features.5.4", # identity_1
        "features.6.6", # identity_2
        "features.7.3", # identity_3
        "identity_global_pool"
        ]])]
    return layer_names_conv + layer_names_identity

def get_features(batch, config, attribution):

    batch.requires_grad = True
    dummy_cond = [{"y": 0} for _ in range(len(batch))]
    record_layer=[config["layer_name"]]
    attr = attribution(batch.to(config["device"]), dummy_cond, record_layer=record_layer)
    if config["cav_mode"] == "cavs_full":
        features = attr.activations[config["layer_name"]]
    else:
        # ViT support
        acts = attr.activations[config["layer_name"]]
        acts = acts if acts.dim() > 2 else acts[..., None, None]
        acts = acts.transpose(1,3).transpose(2,3) if "swin_former" in config["model_name"] else acts
        features = acts.flatten(start_dim=2).max(2)[0]
        # features = attr.activations[config["layer_name"]].flatten(start_dim=2).max(2)[0]
    return features

def get_features_and_relevances(x_batch, config, attribution):
    y_pred = attribution.model(x_batch).argmax(1)
    attribution.model.zero_grad()
    x_batch.requires_grad = True
    cond = [{"y": y} for y in y_pred]
    record_layer=[config["layer_name"]]
    attr = attribution(x_batch.to(config["device"]), cond, record_layer=record_layer)
    acts = attr.activations[config["layer_name"]]
    rels = attr.relevances[config["layer_name"]]
    
    acts = acts if acts.dim() > 2 else acts[..., None, None]
    rels = rels if rels.dim() > 2 else rels[..., None, None]
    acts = acts.transpose(1,3).transpose(2,3) if "swin_former" in config["model_name"] else acts
    rels = rels.transpose(1,3).transpose(2,3) if "swin_former" in config["model_name"] else rels
    
    features = acts.flatten(start_dim=2).max(2)[0].detach()
    rels = rels.flatten(start_dim=2).max(2)[0].detach()
    
    return features, rels