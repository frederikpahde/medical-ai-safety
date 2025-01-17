import torch
import timm
from utils.lrp_canonizers import ResNetCanonizer


def get_resnet50_a1(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_resnet("timm/resnet50.a1_in1k", ckpt_path, pretrained, n_class, device)

def get_resnet50_a2(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_resnet("timm/resnet50.a2_in1k", ckpt_path, pretrained, n_class, device)

def get_resnet50_a3(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_resnet("timm/resnet50.a3_in1k", ckpt_path, pretrained, n_class, device)

def get_resnet18_a1(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_resnet("timm/resnet18.a1_in1k", ckpt_path, pretrained, n_class, device)

def get_resnet18_a2(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_resnet("timm/resnet18.a2_in1k", ckpt_path, pretrained, n_class, device)

def get_resnet18_a3(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_resnet("timm/resnet18.a3_in1k", ckpt_path, pretrained, n_class, device)

def get_resnet18d(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_resnet("timm/resnet18d.ra2_in1k", ckpt_path, pretrained, n_class, device)

def get_resnet50d(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_resnet("timm/resnet50d.a1_in1k", ckpt_path, pretrained, n_class, device)

def get_resnet101d(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_resnet("timm/resnet101d.ra2_in1k", ckpt_path, pretrained, n_class, device)


def get_resnet(model_name, ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)

    if n_class and n_class != 1000:
        num_in = model.fc.in_features
        model.fc = torch.nn.Linear(num_in, n_class, bias=True)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    model.input_identity = torch.nn.Identity()
    model.identity_0 = torch.nn.Identity()
    model.identity_1 = torch.nn.Identity()
    model.identity_2 = torch.nn.Identity()
    model.last_conv = torch.nn.Identity()
    model.forward = _forward.__get__(model)
    model.forward_features = _forward_features.__get__(model)

    return model


def _forward(self, x):
    x = self.input_identity(x)
    x = self.forward_features(x)
    x = self.last_conv(x)
    x = self.forward_head(x)
    return x

def _forward_features(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    if self.grad_checkpointing and not torch.jit.is_scripting():
        raise NotImplementedError()
    else:
        x = self.layer1(x)
        x = self.identity_0(x)
        x = self.layer2(x)
        x = self.identity_1(x)
        x = self.layer3(x)
        x = self.identity_2(x)
        x = self.layer4(x)
    return x


def get_resnet_canonizer():
    return [ResNetCanonizer()]
