import torch
import timm
from utils.lrp_canonizers import RexNetCanonizer


def get_rexnet_100(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_rexnet("rexnet_100", ckpt_path, pretrained, n_class, device)

def get_rexnet_130(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_rexnet("rexnet_130", ckpt_path, pretrained, n_class, device)

def get_rexnet_150(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_rexnet("rexnet_150", ckpt_path, pretrained, n_class, device)


def get_rexnet(model_name, ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)

    if n_class and n_class != 1000:
        num_in = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(num_in, n_class, bias=True)
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
    model.stem_identity = torch.nn.Identity()
    
    for i in range(len(model.features) - 1):
        print("added identity", i)
        setattr(model, f"identity_{i}", torch.nn.Identity())

    model.last_conv = torch.nn.Identity()
    model.forward_features = _forward_features.__get__(model)

    return model


def _forward_features(self, x):
    x = self.input_identity(x)
    x = self.stem(x)
    x = self.stem_identity(x)
    if self.grad_checkpointing and not torch.jit.is_scripting():
        raise NotImplementedError()
    else:
        for i in range(len(self.features)):
            x = self.features[i](x)
            if hasattr(self, f"identity_{i}"):
                x = getattr(self, f"identity_{i}")(x)
    x = self.last_conv(x)
    return x


def get_rexnet_canonizer():
    return [RexNetCanonizer()]
