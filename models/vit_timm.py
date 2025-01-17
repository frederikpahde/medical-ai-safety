import torch
import timm

from utils.lrp_canonizers import VITCanonizer

def get_vit_b_16_google(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_vit("google/vit_base_patch16_224", ckpt_path, pretrained, n_class, device)

def get_vit_b_16_1k(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_vit("vit_base_patch16_224.augreg_in1k", ckpt_path, pretrained, n_class, device)

def get_vit_b_16_21k(ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    return get_vit("vit_base_patch16_224.augreg_in21k", ckpt_path, pretrained, n_class, device)

def get_vit(model_name, ckpt_path=None, pretrained=True, n_class: int = None, device="cuda") -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)

    if n_class and n_class != 1000:
        model.head = torch.nn.Linear(model.head.in_features, n_class)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    model.inspection_layer = torch.nn.Identity()
    for i in range(len(model.blocks)):
        print("added identity", i)
        setattr(model, f"identity_{i}", torch.nn.Identity())

    model.forward_head = _forward_head.__get__(model)
    model.forward_features = _forward_features.__get__(model)

    return model

def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            raise NotImplementedError
        else:
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
                if hasattr(self, f"identity_{i}"):
                    x = getattr(self, f"identity_{i}")(x)
            # x = self.blocks(x)
        x = self.norm(x)
        return x

def _forward_head(self, x, pre_logits=False):
    if self.attn_pool is not None:
        x = self.attn_pool(x)
    elif self.global_pool == 'avg':
        x = x[:, self.num_prefix_tokens:].mean(dim=1)
    elif self.global_pool:
        x = x[:, 0]  # class token
    x = self.inspection_layer(x)
    x = self.fc_norm(x)
    x = self.head_drop(x)
    return x if pre_logits else self.head(x)

def get_vit_canonizer():
    return [VITCanonizer()]
