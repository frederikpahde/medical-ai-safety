import copy
import numpy as np
import torch
from utils.helper import get_features
import torchvision.transforms as T
from skimage.filters import threshold_otsu

def get_localizations(x, cav, attribution, composite, config, device):
    _config = copy.deepcopy(config)
    _config["cav_mode"] = "cavs_full"
    act = get_features(x.to(device), _config, attribution).detach()
    init_rel = (act.clamp(min=0) * cav[..., None, None].to(device)).to(device)
    attr = attribution(x.to(device).requires_grad_(), [{}], composite, start_layer=config["layer_name"], init_rel=init_rel)
    hms = attr.heatmap.detach().cpu().clamp(min=0)
    return attr, hms

def binarize_heatmaps(hms, kernel_size=7, sigma=8.0, thresholding="otsu", percentile=92):
    gaussian = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    heatmaps_binary = []
    for hm in hms:
        hm_smooth = gaussian(hm.clamp(min=0)[None])[0].numpy()
        if thresholding == "otsu":
            thresh = threshold_otsu(hm_smooth)
        else:
            thresh = np.percentile(hm_smooth, percentile)
        heatmaps_binary.append((hm_smooth > thresh).astype(np.uint8))
    return torch.Tensor(np.array(heatmaps_binary)).type(torch.uint8)