import torch
import torch.hub
from torch import Tensor
from torchvision.models import efficientnet_b0, efficientnet_b4, efficientnet_v2_s

from utils.lrp_canonizers import EfficientNetBNCanonizer


def get_efficientnet_b0(ckpt_path=None, pretrained=True, device="cuda", n_class: int = None) -> torch.nn.Module:
    return get_efficientnet(efficientnet_b0, ckpt_path, pretrained, n_class, device=device)

def get_efficientnet_b0_maxpool(ckpt_path=None, pretrained=True, device="cuda", n_class: int = None) -> torch.nn.Module:
    return get_efficientnet(efficientnet_b0, ckpt_path, pretrained, n_class, pool="max", device=device)


def get_efficientnet_b4(ckpt_path=None, pretrained=True, device="cuda", n_class: int = None) -> torch.nn.Module:
    return get_efficientnet(efficientnet_b4, ckpt_path, pretrained, n_class, device=device)

def get_efficientnet_b4_maxpool(ckpt_path=None, pretrained=True, device="cuda", n_class: int = None) -> torch.nn.Module:
    return get_efficientnet(efficientnet_b4, ckpt_path, pretrained, pool="max", device=device, n_class=n_class)

def get_efficientnet_v2_s(ckpt_path=None, pretrained=True, device="cuda", n_class: int = None) -> torch.nn.Module:
    return get_efficientnet(efficientnet_v2_s, ckpt_path, pretrained, n_class, device=device)

def get_efficientnet(model_fn, ckpt_path=None, pretrained=True, device="cuda", n_class: int = None, pool="avg") -> torch.nn.Module:
    print(f"In get_efficitentnet: {ckpt_path}")
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None

    model = model_fn(weights=weights)

    if n_class:
        classifier = list(model.classifier.children())
        new_layer = classifier[-1] if n_class == 1000 else torch.nn.Linear(classifier[-1].in_features, n_class)
        model.classifier = torch.nn.Sequential(*classifier[:-1])
        model.classifier.add_module('last', new_layer)
    if ckpt_path:
        print("Loading from existing checkpoint.")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint)

    for i in range(len(model.features) - 1):
        print("added identity", i)
        setattr(model, f"identity_{i}", torch.nn.Identity())
    model.input_identity = torch.nn.Identity()
    model.last_conv = torch.nn.Identity()
    model.last_relu = torch.nn.ReLU(inplace=False)
    model._forward_impl = _forward_impl_modified.__get__(model)
    model.features[-1][2] = torch.nn.Sequential(*[torch.nn.Identity(), torch.nn.SiLU(inplace=False)])
    model.maxpool = torch.nn.AdaptiveMaxPool2d(1)
    model.fn_pool = pool
    return model


def _forward_impl_modified(self, x: Tensor) -> Tensor:
    x = self.input_identity(x)
    for i in range(len(self.features)):
        x = self.features[i](x)
        if hasattr(self, f"identity_{i}"):
            x = getattr(self, f"identity_{i}")(x)

    x = self.last_relu(self.last_conv(x))  # added identity

    if self.fn_pool == "avg":
        x = self.avgpool(x)
    elif self.fn_pool == "max":
        x = self.maxpool(x)
    else:
        raise ValueError(f"Unknown pool {self.fn_pool}")
    x = torch.flatten(x, 1)

    x = self.classifier(x)

    return x


def get_efficientnet_canonizer():
    return [EfficientNetBNCanonizer()]
