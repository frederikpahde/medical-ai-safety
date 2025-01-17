import logging
from typing import Callable

from datasets.chexpert.chexpert_pm import NORM_PARAMS_CHEXPERT, CheXpertPMDataset, get_chexpert_pm_dataset
from datasets.chexpert.chexpert_pm_attacked import CheXpertPMAttackedDataset, get_chexpert_pm_attacked_dataset
from datasets.chexpert.chexpert_pm_attacked_hm import CheXpertPMAttackedHmDataset, get_chexpert_pm_attacked_hm_dataset
from datasets.isic.isic24 import NORM_PARAMS_ISIC24, ISIC24Dataset, get_isic24_dataset
from datasets.isic.isic import NORM_PARAMS_ISIC, ISICDataset, get_isic_dataset
from datasets.isic.isic_attacked import get_isic_attacked_dataset
from datasets.isic.isic_attacked_hm import get_isic_attacked_hm_dataset
from datasets.isic.isic_hm import get_isic_hm_dataset
from datasets.kvasir.hyper_kvasir_attacked import get_hyper_kvasir_attacked_dataset
from datasets.kvasir.hyper_kvasir_attacked_hm import get_hyper_kvasir_attacked_hm_dataset
from datasets.kvasir.hyper_kvasir import NORM_PARAMS_KVASIR, HyperKvasirDataset, get_hyper_kvasir_dataset
from datasets.kvasir.hyper_kvasir_hm import HyperKvasirHmDataset, get_hyper_kvasir_hm_dataset
from datasets.ecg_ptb_xl.ptb_xl import PtbXlDataset, get_ptb_xl_dataset
from datasets.ecg_ptb_xl.ptb_xl_attacked import PtbXlAttackedDataset, get_ptb_xl_attacked_dataset
from datasets.ecg_ptb_xl.ptb_xl_attacked_hm import PtbXlAttackedHmDataset, get_ptb_xl_attacked_hm_dataset
from utils.artificial_artifact import get_artifact_kwargs

logger = logging.getLogger(__name__)

DATASETS = {
    "isic24": get_isic24_dataset,
    "isic": get_isic_dataset,
    "isic_hm": get_isic_hm_dataset,
    "isic_attacked": get_isic_attacked_dataset,
    "isic_attacked_hm": get_isic_attacked_hm_dataset,
    "hyper_kvasir": get_hyper_kvasir_dataset,
    "hyper_kvasir_hm": get_hyper_kvasir_hm_dataset,
    "hyper_kvasir_attacked": get_hyper_kvasir_attacked_dataset,
    "hyper_kvasir_attacked_hm": get_hyper_kvasir_attacked_hm_dataset,
    "chexpert_pm_attacked": get_chexpert_pm_attacked_dataset,
    "chexpert_pm_attacked_hm": get_chexpert_pm_attacked_hm_dataset,
    "chexpert_pm": get_chexpert_pm_dataset,
    "ptb_xl": get_ptb_xl_dataset,
    "ptb_xl_attacked": get_ptb_xl_attacked_dataset,
    "ptb_xl_attacked_hm": get_ptb_xl_attacked_hm_dataset,
}

DATASET_CLASSES = {
    "isic24": ISIC24Dataset,
    "isic": ISICDataset,
    "isic_attacked": ISICDataset,
    "hyper_kvasir": HyperKvasirDataset,
    "hyper_kvasir_hm": HyperKvasirHmDataset,
    "hyper_kvasir_attacked": HyperKvasirDataset,
    "hyper_kvasir_attacked_hm": HyperKvasirDataset,
    "chexpert_pm": CheXpertPMDataset,
    "chexpert_pm_attacked": CheXpertPMAttackedDataset,
    "chexpert_pm_attacked_hm": CheXpertPMAttackedHmDataset,
    "ptb_xl": PtbXlDataset,
    "ptb_xl_attacked": PtbXlAttackedDataset,
    "ptb_xl_attacked_hm": PtbXlAttackedHmDataset,
}

DATASET_NORM_PARAMS = {
    # (means, vars)
    "isic": NORM_PARAMS_ISIC,
    "isic_attacked": NORM_PARAMS_ISIC,
    "isic24": NORM_PARAMS_ISIC24,
    "hyper_kvasir": NORM_PARAMS_KVASIR,
    "hyper_kvasir_hm": NORM_PARAMS_KVASIR,
    "hyper_kvasir_attacked": NORM_PARAMS_KVASIR,
    "hyper_kvasir_attacked_hm": NORM_PARAMS_KVASIR,
    "chexpert_pm": NORM_PARAMS_CHEXPERT,
    "chexpert_pm_attacked": NORM_PARAMS_CHEXPERT,
    "chexpert_pm_attacked_hm": NORM_PARAMS_CHEXPERT,
}


def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.

    """
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        logger.info(f"Loading {dataset_name}")
        return dataset
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")
    
def get_dataset_kwargs(config):
    dataset_specific_kwargs = {
        "label_map_path": config["label_map_path"],
        "classes": config.get("classes", None),
        "train": True
    } if "imagenet" in config['dataset_name'] else {}

    return dataset_specific_kwargs

def load_dataset(config, normalize_data=True, hm=False):
    dataset_name = config['dataset_name']
    dataset_name = f"{dataset_name}_hm" if hm else dataset_name
    data_paths = config['data_paths']
    img_size = config.get("img_size", 224)
    binary_target = config.get('binary_target', None)
    attacked_classes = config.get("attacked_classes", [])
    p_artifact = config.get("p_artifact", None)
    artifact_type = config.get("artifact_type", None)
    artifact_ids_file=config.get('artifacts_file', None)
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)
    if hm:
        if "attacked" in dataset_name:
            # specify whether to use GT or predicted masks
            source_maks = config.get("source_maks", "gt")
            assert source_maks in ["gt", "hm", "bin"], f"Unknown mask source: {source_maks}"
            dataset_specific_kwargs["source_masks"] = source_maks

        else:
            # specify artifact
            dataset_specific_kwargs["artifact"] = config["artifact"]

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=normalize_data,
                                        image_size=img_size,
                                        binary_target=binary_target,
                                        attacked_classes=attacked_classes,
                                        p_artifact=p_artifact,
                                        artifact_type=artifact_type,
                                        artifact_ids_file=artifact_ids_file,
                                        **artifact_kwargs, **dataset_specific_kwargs)
    return dataset