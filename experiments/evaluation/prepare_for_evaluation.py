import torch
from torch.utils.data import Dataset
from model_correction import get_correction_method

def prepare_model_for_evaluation(
        model: torch.nn.Module,
        dataset: Dataset,
        device: str,
        config: dict) -> torch.nn.Module:
    """ 
    Prepare corrected model for evaluation. Brings model to eval-mode and to the desired device.
    For P-ClArC methods (weights remain unchanged), the projection hook is added to the model.

    Args:
        model (torch.nn.Module): Model
        dataset (Dataset): Train Dataset
        device (str): device name
        config (dict): config

    Returns:
        torch.nn.Module: Model to be evaluated
    """

    method = config['method']
    kwargs_correction = {}
    correction_method = get_correction_method(method)

    if "pclarc" in method.lower():
        mode = "cavs_max"
        kwargs_correction['classes'] = dataset.classes
        kwargs_correction['artifact_sample_ids'] = dataset.sample_ids_by_artifact[config['artifact']]
        kwargs_correction['sample_ids'] = dataset.idxs_train
        kwargs_correction['mode'] = mode
        kwargs_correction["eval_mode"] = True

        model = correction_method(model, config, **kwargs_correction)

    model = model.to(device)
    model.eval()
    return model
