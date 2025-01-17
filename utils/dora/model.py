from torchvision.models import VGG, ResNet
from timm.models import ResNet as ResNetTimm
import torch

class DoraModel(torch.nn.Module):
    def __init__(self, ft_extractor, pool):
        super().__init__()
        self.ft_extractor = ft_extractor
        self.pool = pool

    def forward(self, x):
        x = self.ft_extractor(x)
        x = self.pool(x)
        return x

def get_dim(model, input_size, device):
    x = torch.rand(1, 3, input_size, input_size).to(device)
    out = model(x)
    n_dim = out.shape[1]
    return n_dim

def modify_model(model, layer_name, aggr="max"):
    if isinstance(model, VGG):
        feature_extractor = get_ft_extractor_vgg(model, layer_name)
    elif isinstance(model, ResNetTimm):
        feature_extractor = get_ft_extractor_resnet_timm(model, layer_name)
    elif isinstance(model, ResNet):
        feature_extractor = get_ft_extractor_resnet(model, layer_name)
    return DoraModel(feature_extractor, get_aggregator(aggr))

def get_aggregator(aggr):
    if aggr == "max":
        return torch.nn.AdaptiveAvgPool2d(1)
    elif aggr == "avg":
        return torch.nn.AdaptiveAvgPool2d(1)
    else:
        raise ValueError(f"Unknown aggregator: {aggr}, use one in (max,avg)")
    
def get_ft_extractor_vgg(model, layer_name):
    if layer_name.startswith("features"):
        idx = int(layer_name.split(".")[-1])
    else:
        raise ValueError(f"Unknown layer name for VGG: {layer_name}")
    _model = model.features[:idx+1]
    return _model

def get_ft_extractor_resnet(model, layer_name):
    def _forward_resnet(self, x):
        x = self.input_identity(x)
        if layer_name == "input_identity":
            return x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.identity_0(x)  # added identity
        if layer_name == "identity_0":
            return x
        x = self.relu_0(x)

        x = self.layer2(x)
        x = self.identity_1(x)  # added identity
        if layer_name == "identity_1":
            return x
        x = self.relu_1(x)

        x = self.layer3(x)
        x = self.identity_2(x)  # added identity
        if layer_name == "identity_2":
            return x
        x = self.relu_2(x)

        x = self.layer4(x)
        x = self.last_conv(x)
        if layer_name == "last_conv":
            return x
        raise ValueError(f"Unknown layer name for ResNet: {layer_name}")
    model._forward_impl = _forward_resnet.__get__(model)
    return model

def get_ft_extractor_resnet_timm(model, layer_name):

    def _forward_features_timm(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            raise NotImplementedError()
        else:
            x = self.layer1(x)
            x = self.identity_0(x)
            if layer_name == "identity_0":
                return x
            x = self.layer2(x)
            x = self.identity_1(x)
            if layer_name == "identity_1":
                return x
            x = self.layer3(x)
            x = self.identity_2(x)
            if layer_name == "identity_2":
                return x
            x = self.layer4(x)
            x = self.last_conv(x)
            if layer_name == "last_conv":
                return x
            raise ValueError(f"Unknown layer name for ResNet: {layer_name}")
        
    model.forward = _forward_features_timm.__get__(model)
    return model