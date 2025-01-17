import logging

import torch

from models.efficientnet import (get_efficientnet_b0, 
                                 get_efficientnet_b0_maxpool,
                                 get_efficientnet_b4, 
                                 get_efficientnet_b4_maxpool, 
                                 get_efficientnet_v2_s,
                                 get_efficientnet_canonizer)

from models.resnet_timm import (get_resnet18_a1,
                           get_resnet18_a2,
                           get_resnet18_a3,
                           get_resnet18d,
                           get_resnet50_a1,
                           get_resnet50_a2,
                           get_resnet50_a3,
                           get_resnet50d,
                           get_resnet101d,
                           get_resnet_canonizer)
from models.resnet import get_resnet18, get_resnet101
from models.rexnet import get_rexnet_100, get_rexnet_130, get_rexnet_150, get_rexnet_canonizer
from models.vgg import get_vgg16, get_vgg16_bn, get_vgg11, get_vgg11_bn, get_vgg13_bn, get_vgg13, get_vgg_canonizer
from models.vit_timm import (get_vit_b_16_1k, 
                        get_vit_b_16_21k, 
                        get_vit_b_16_google,
                        get_vit_canonizer)

from models.resnext import (get_resnext50,
                            get_resnext_canonizer)

from models.densenet import (get_densenet121, get_densenet169, get_densenet_canonizer)

from models.xresnet1d import (get_xresnet1d50, get_xresnet1d50_canonizer)

from models.vit import (get_vit_b_16 as get_vit_b_16_torchvision, 
                        get_vit_canonizer as get_vit_canonizer_torchvision)

logger = logging.getLogger(__name__)

TRANSFORMER_MODELS = [
    "vit",
    ]

MODELS_1D = [
    "vit", 
    ]

MODELS = {
    "vgg16": get_vgg16,
    "vgg16_bn": get_vgg16_bn,
    "vgg13": get_vgg13,
    "vgg13_bn": get_vgg13_bn,
    "vgg11": get_vgg11,
    "vgg11_bn": get_vgg11_bn,
    "efficientnet_b0": get_efficientnet_b0,
    "efficientnet_b0_avgpool": get_efficientnet_b0,
    "efficientnet_b0_maxpool": get_efficientnet_b0_maxpool,
    "efficientnet_b4": get_efficientnet_b4,
    "efficientnet_b4_avgpool": get_efficientnet_b4,
    "efficientnet_b4_maxpool": get_efficientnet_b4_maxpool,
    "efficientnet_v2_s": get_efficientnet_v2_s,
    "densenet121": get_densenet121,
    "densenet169": get_densenet169,
    "resnet18": get_resnet18,
    "resnet101": get_resnet101,
    "resnet101d": get_resnet101d,
    "resnet18d": get_resnet18d,
    "resnet18_a1": get_resnet18_a1,
    "resnet18_a2": get_resnet18_a2,
    "resnet18_a3": get_resnet18_a3,
    "resnet50d": get_resnet50d,
    "resnet50_a1": get_resnet50_a1,
    "resnet50_a2": get_resnet50_a2,
    "resnet50_a3": get_resnet50_a3,

    "resnext50": get_resnext50,

    "vit_b_16_google": get_vit_b_16_google,
    "vit_b_16_1k": get_vit_b_16_1k,
    "vit_b_16_21k": get_vit_b_16_21k,
    "vit_b_16_torchvision": get_vit_b_16_torchvision,

    "rexnet_100": get_rexnet_100,
    "rexnet_130": get_rexnet_130,
    "rexnet_150": get_rexnet_150,
    
    "xresnet1d50": get_xresnet1d50
}

CANONIZERS = {
    "vgg16": get_vgg_canonizer,
    "vgg16_bn": get_vgg_canonizer,
    "vgg13": get_vgg_canonizer,
    "vgg13_bn": get_vgg_canonizer,
    "vgg11": get_vgg_canonizer,
    "vgg11_bn": get_vgg_canonizer,
    "efficientnet_b0": get_efficientnet_canonizer,
    "efficientnet_b0_avgpool": get_efficientnet_canonizer,
    "efficientnet_b0_maxpool": get_efficientnet_canonizer,
    "efficientnet_b4": get_efficientnet_canonizer,
    "efficientnet_b4_avgpool": get_efficientnet_canonizer,
    "efficientnet_b4_maxpool": get_efficientnet_canonizer,
    "efficientnet_v2_s": get_efficientnet_canonizer,
    "densenet121": get_densenet_canonizer,
    "densenet169": get_densenet_canonizer,
    "resnet18": get_resnet_canonizer,
    "resnet101": get_resnet_canonizer,
    "resnet101d": get_resnet_canonizer,
    "resnet18d": get_resnet_canonizer,
    "resnet18_a1": get_resnet_canonizer,
    "resnet18_a2": get_resnet_canonizer,
    "resnet18_a3": get_resnet_canonizer,
    "resnet50d": get_resnet_canonizer,
    "resnet50_a1": get_resnet_canonizer,
    "resnet50_a2": get_resnet_canonizer,
    "resnet50_a3": get_resnet_canonizer,
    "resnext50": get_resnext_canonizer,
    

    "rexnet_100": get_rexnet_canonizer,
    "rexnet_130": get_rexnet_canonizer,
    "rexnet_150": get_rexnet_canonizer,

    "vit_b_16_google": get_vit_canonizer,
    "vit_b_16_1k": get_vit_canonizer,
    "vit_b_16_21k": get_vit_canonizer,
    "vit_b_16_torchvision": get_vit_canonizer_torchvision,

    "xresnet1d50": get_xresnet1d50_canonizer,
    # "xresnet1d50": None
}


def get_canonizer(model_name):
    assert model_name in list(CANONIZERS.keys()), f"No canonizer for model '{model_name}' available"
    return CANONIZERS[model_name]()


def get_fn_model_loader(model_name: str) -> torch.nn.Module:
    if model_name in MODELS:
        fn_model_loader = MODELS[model_name]
        logger.info(f"Loading {model_name}")
        return fn_model_loader
    else:
        raise KeyError(f"Model {model_name} not available")