from torchvision.models import ResNet, EfficientNet, VGG, VisionTransformer
from timm.models.resnet import ResNet as ResNetTimm
from timm.models.rexnet import RexNet
from timm.models.vision_transformer import VisionTransformer as VisionTransformerTimm
from models.xresnet1d import ECGResNet

LAYER_NAME_MAP = {
    "resnet18": "resnet",
    "resnet50": "resnet",
    "resnet50d": "resnet",
    "resnet101d": "resnet",
    "vit_b_16_torchvision": "vit"
}

LAYER_NAMES_BY_MODEL = {
    'vgg11': [
        "input_identity", 
        "features.0",
        "features.3",
        "features.6",
        "features.8",
        "features.11",
        "features.13",
        "features.16",
        "features.18"
        ],
    'vgg16': [
        "input_identity", 
        # "features.1",  # Conv 1
        "features.3",  # Conv 2
        "features.6",  # Conv 3
        "features.8",  # Conv 4
        "features.11", # Conv 5
        "features.13", # Conv 6
        "features.15", # Conv 7
        "features.18", # Conv 8
        "features.20", # Conv 9
        "features.22", # Conv 10      
        "features.25", # Conv 11
        "features.27", # Conv 12
        # "features.28", # Conv 12
        "features.29"  # Conv 13
        ],
    'vgg16_with_relu': [
        "features.0",
        "features.1",
        "features.2",
        "features.3",
        "features.5",
        "features.6",
        "features.7",
        "features.8",
        "features.10",
        "features.11",
        "features.12",
        "features.13",
        "features.14",
        "features.15",
        "features.17",
        "features.18",
        "features.19",
        "features.20",
        "features.21",
        "features.22",
        "features.24",
        "features.25",
        "features.26",
        "features.27",
        "features.28",
        "features.29"
        ],
    'resnet': [
        "input_identity", 
        "identity_0",
        "identity_1",
        "identity_2",
        "last_conv"
        ],
    'xresnet1d50': [
        "features.4.3", # identity_0
        "features.5.4", # identity_1
        "features.6.6", # identity_2
        "features.7.3", # identity_3
        "identity_global_pool"
    ],
    'resnext': [
        "input_identity", 
        "identity_0",
        "identity_1",
        "identity_2",
        "last_conv"
        ],
    'rexnet': [
        "input_identity", 
        "identity_12",
        "identity_13",
        "identity_14",
        "identity_15",
        "last_conv"
        ],
    'efficientnet_b0': [
        # "input_identity", 
        # "identity_0",
        # "identity_1",
        # "identity_2",
        # "identity_3",
        # "identity_4",
        "identity_5",
        "identity_6",
        "identity_7",
        "last_conv"
        ],
        'efficientnet_v2': [
        "input_identity", 
        "identity_0",
        "identity_1",
        "identity_2",
        "identity_3",
        "identity_4",
        "identity_5",
        "identity_6",
        "last_conv"
        ],
        "vit": [
            # "identity_8",
            # "identity_9",
            # "identity_10",
            "inspection_layer"
        ],
        "swin_former": [
            # "identity_0",
            # "identity_1",
            # "identity_2",
            "inspection_layer"
        ],
        "metaformer": [
            # "identity_0",
            # "identity_1",
            # "identity_2",
            "inspection_layer"
        ],
        "densenet121":[
            "features.transition1.identity",
            "features.transition2.identity",
            "features.transition3.identity",
            "features.last_conv"
        ]
}

def get_lnames_sorted_resnet(model):
    lnames = [n for n, _ in model.named_modules()]
    lnames_sorted = ["input_identity"]
    idx = 0
    while "layer" not in lnames[idx]:
        lnames_sorted.append(lnames[idx])
        idx += 1

    for lidx in range(4):
        while f"layer{lidx+1}" in lnames[idx]:
            lnames_sorted.append(lnames[idx])
            idx += 1
        name_identity = "last_conv" if lidx == 3 else f"identity_{lidx}"
        name_relu = "last_relu" if lidx == 3 else f"relu_{lidx}"
        lnames_sorted += [name_identity, name_relu]

    while idx < len(lnames):
        if "identity" not in lnames[idx] and "last_" not in lnames[idx]:
            lnames_sorted.append(lnames[idx])
        idx += 1
    return lnames_sorted

def get_lnames_sorted_rexnet(model):
    lnames = [n for n, _ in model.named_modules()]
    lnames_sorted = ["input_identity"]
    idx = 0
    while "features." not in lnames[idx]:
        lnames_sorted.append(lnames[idx])
        idx += 1
    lnames_sorted.append("stem_identity")
    for lidx in range(17):
        while f"features.{lidx}" in lnames[idx]:
            lnames_sorted.append(lnames[idx])
            idx += 1
        name_identity = "last_conv" if lidx == 16 else f"identity_{lidx}"
        name_relu = "last_relu" if lidx == 16 else f"relu_{lidx}"
        lnames_sorted += [name_identity, name_relu]

    while idx < len(lnames):
        if "identity" not in lnames[idx] and "last_" not in lnames[idx]:
            lnames_sorted.append(lnames[idx])
        idx += 1
    return lnames_sorted

def get_lnames_sorted_efficientnet(model):
    lnames = [n for n, _ in model.named_modules()]
    lnames_sorted = ["input_identity"]
    idx = 0
    num_blocks = max([int(n.split(".")[1]) for n in lnames if "features." in n])

    while not "features.0" in lnames[idx]:
        lnames_sorted.append(lnames[idx])
        idx += 1

    for lidx in range(num_blocks + 1):
        while f"features.{lidx}" in lnames[idx]:
            lnames_sorted.append(lnames[idx])
            idx += 1
        name_identity = "last_conv" if lidx == num_blocks else f"identity_{lidx}"
        name_relu = "last_relu" if lidx == num_blocks else f"relu_{lidx}"
        lnames_sorted += [name_identity, name_relu]

    while idx < len(lnames):
        if "identity" not in lnames[idx] and "last_" not in lnames[idx] and "stem" not in lnames[idx]:
            lnames_sorted.append(lnames[idx])
        idx += 1
        
    return lnames_sorted

def get_lnames_sorted_vgg(model):
    return ["input_identity"] + [n for n, _ in model.named_modules()][:-1]

def get_lnames_sorted_vit(model):
    return [n for n, _ in model.named_modules()][:-3] + ["inspection_layer"] + [n for n, _ in model.named_modules()][-3:-1]

def get_lnames_sorted_vit_timm(model):
    lnames = [n for n, _ in model.named_modules()]
    lnames_sorted = []
    idx = 0
    while "blocks." not in lnames[idx]:
        lnames_sorted.append(lnames[idx])
        idx += 1

    for lidx in range(12):
        while f"blocks.{lidx}" in lnames[idx]:
            lnames_sorted.append(lnames[idx])
            idx += 1
        name_identity = f"identity_{lidx}"
        name_relu = f"relu_{lidx}"
        lnames_sorted += [name_identity, name_relu]

    while idx < len(lnames):
        if lnames[idx] == "head":
            lnames_sorted.append("inspection_layer")
        if not any([l in lnames[idx] for l in ["identity", "inspection"]]):
            lnames_sorted.append(lnames[idx])
        idx += 1
    return lnames_sorted

def get_lnames_sorted_ecg_resnet(model):
    return [n for n, _ in model.named_modules()]

def get_lnames_sorted(model):
    if isinstance(model, ResNet):
        return get_lnames_sorted_resnet(model)
    elif isinstance(model, ResNetTimm):
        return get_lnames_sorted_resnet(model)
    elif isinstance(model, RexNet):
        return get_lnames_sorted_rexnet(model)
    elif isinstance(model, EfficientNet):
        return get_lnames_sorted_efficientnet(model)
    elif isinstance(model, VGG):
        return get_lnames_sorted_vgg(model)
    elif isinstance(model, VisionTransformer):
        return get_lnames_sorted_vit(model)
    elif isinstance(model, VisionTransformerTimm):
        return get_lnames_sorted_vit_timm(model)
    elif isinstance(model, ECGResNet):
        return get_lnames_sorted_ecg_resnet(model)
    else:
        raise NotImplementedError(f"not implemented for model {model.__class__}")