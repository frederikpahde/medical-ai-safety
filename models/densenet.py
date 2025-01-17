import torch
import torch.hub
from torchvision.models import densenet121, densenet169

from utils.lrp_canonizers import SequentialThreshCanonizer


def get_densenet121(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_densenet(densenet121, ckpt_path, pretrained, n_class, device)

def get_densenet169(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_densenet(densenet169, ckpt_path, pretrained, n_class, device)

def get_densenet(model_fn, ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None
    model = model_fn(weights=weights)
    if n_class != 1000:
        num_ftrs = model.classifier.in_features  
        model.classifier = torch.nn.Linear(num_ftrs, n_class)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        model.load_state_dict(checkpoint)

    model.last_conv = torch.nn.Identity()
    model.last_relu = torch.nn.ReLU()
    model.pool = torch.nn.AdaptiveAvgPool2d((1,1))
    model.forward = _forward_densenet.__get__(model)
    return model


def get_densenet_canonizer():
    return [SequentialThreshCanonizer()]

def _forward_densenet(self, x):
    features = self.features(x)
    out = self.last_conv(features)
    out = self.last_relu(out)
    out = self.pool(out)
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out