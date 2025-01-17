import torch
import torch.hub
from torchvision.models import vgg16, vgg16_bn, vgg11, vgg13, vgg13_bn, vgg11_bn
from zennit.torchvision import VGGCanonizer


def get_vgg16(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_vgg(vgg16, ckpt_path, pretrained, n_class, device)


def get_vgg16_bn(ckpt_path=None, pretrained=True, n_class=None, device="cuda") -> torch.nn.Module:
    return get_vgg(vgg16_bn, ckpt_path, pretrained, n_class, device)


def get_vgg13(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_vgg(vgg13, ckpt_path, pretrained, n_class, device)


def get_vgg13_bn(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_vgg(vgg13_bn, ckpt_path, pretrained, n_class, device)


def get_vgg11(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_vgg(vgg11, ckpt_path, pretrained, n_class, device)


def get_vgg11_bn(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_vgg(vgg11_bn, ckpt_path, pretrained, n_class, device)


def get_vgg(model_fn, ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None

    model = model_fn(weights=weights)

    if n_class and n_class != 1000:
        model.classifier[-1] = torch.nn.Linear(4096, n_class, bias=True)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("classifier.last", "classifier.6"): v for k, v in checkpoint.items()}  # ISIC MODEL
        model.load_state_dict(checkpoint)
    model.input_identity = torch.nn.Identity()
    model.forward = forward_modified.__get__(model)
    return model


def get_vgg_canonizer():
    return [VGGCanonizer()]

def forward_modified(self, x: torch.Tensor) -> torch.Tensor:
    x = self.input_identity(x)
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x