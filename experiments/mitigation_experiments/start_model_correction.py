import logging
import os
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasets import load_dataset
from experiments.evaluation.evaluate_by_subset import evaluate_by_subset
from experiments.evaluation.evaluate_by_subset_attacked import evaluate_by_subset_attacked
from model_correction import get_correction_method
from models import get_fn_model_loader
from utils.helper import load_config

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

def get_parser():
    parser = ArgumentParser(
        description='Run bias mitigation experiments.', )
    parser.add_argument('--num_gpu', default=1)
    parser.add_argument('--config_file', default="config_files/bias_mitigation_controlled/hyper_kvasir_attacked/local/resnet50d_RRClarc_lamb1000000_adam_lr0.001_identity_2.yaml")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    config = load_config(args.config_file)
    start_model_correction(config, args.num_gpu)


def start_model_correction(config, num_gpu):
    """ Starts model correction for given config file.

    Args:
        config (dict): Dictionary with config parameters for training.
    """

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    config_name = config["config_name"]
    logger.info(f"Running {config_name}")

    dataset_name = config['dataset_name']

    model_name = config['model_name']
    ckpt_path = config['ckpt_path']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    optimizer_name = config['optimizer']
    loss_name = config['loss']
    checkpoint_dir_corrected = config['checkpoint_dir_corrected']
    lr = config['lr']
    limit_train_batches = config.get("limit_train_batches", None)
    method = config["method"]
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config.get('device', default_device)
    wandb_project_name = config.get('wandb_project_name', None)
    wandb_api_key = config.get('wandb_api_key', None)

    do_wandb_logging = wandb_project_name is not None

    logger_ = None
    # Initialize WandB
    if do_wandb_logging:
        assert wandb_api_key is not None, f"'wandb_api_key' required if 'wandb_project_name' is provided ({wandb_project_name})"
        os.environ["WANDB_API_KEY"] = wandb_api_key
        logger.info(f"Initialized wand. Logging to {wandb_project_name} / {config_name}...")
        wandb_id = f"{config_name}" if config.get('unique_wandb_ids', True) else None
        logger_ = WandbLogger(project=wandb_project_name, name=f"{config_name}", id=wandb_id, config=config)

    # Load Data
    require_masks = "rrr" in method.lower()
    dataset = load_dataset(config, normalize_data=True, hm=require_masks)

    # Load Model
    fn_model_loader = get_fn_model_loader(model_name)

    model = fn_model_loader(
        ckpt_path=ckpt_path,
        n_class=len(dataset.classes),
        device=device).to(device)

    # Construct correction kwargs
    kwargs_correction = {}
    if "clarc" in method.lower():
        kwargs_correction['classes'] = dataset.classes
        kwargs_correction['artifact_sample_ids'] = dataset.sample_ids_by_artifact[config['artifact']]
        kwargs_correction['sample_ids'] = np.array([i for i in dataset.idxs_train]) 
        kwargs_correction['mode'] = config["cav_mode"]

    # Define Optimizer and Loss function
    correction_method = get_correction_method(method)
    model_corrected = correction_method(model, config, **kwargs_correction)

    # Define Optimizer and Loss function
    model_corrected.set_optimizer(optimizer_name, model_corrected.parameters(), lr, ckpt_path)

    weights = None if dataset_name == "imagenet" else dataset.weights
    model_corrected.set_loss(loss_name, weights=weights)
        
    # Split data into train/val
    idxs_train = dataset.idxs_train
    idxs_val = dataset.idxs_val

    dataset_train = dataset.get_subset_by_idxs(idxs_train)
    dataset_val = dataset.get_subset_by_idxs(idxs_val)

    dataset_train.do_augmentation = True  

    logger.info(f"Number of samples: {len(dataset_train)} (train) / {len(dataset_val)} (val)")

    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    checkpoint_callback = ModelCheckpoint(monitor="valid_acc",
                                          dirpath=f"{checkpoint_dir_corrected}/{config_name}",
                                          filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
                                          auto_insert_metric_name=False,
                                          save_last=True,
                                          save_weights_only=True,
                                          mode="max")

    timer = Timer()

    class EvalBySubset(Callback):
        def on_train_start(self, trainer, pl_module):
            print("Training is starting")

        def on_train_end(self, trainer, pl_module):
            print("Training is ending")

        def on_train_epoch_start(self, trainer, pl_module):
            print("Saving checkpoint")
            if trainer.current_epoch >= 1:
                if "celeba" in dataset_name:
                    evaluate_by_subset(config)
                else:
                    evaluate_by_subset_attacked(config)

    trainer = Trainer(callbacks=[
        EvalBySubset() if config.get("eval_acc_every_epoch", False) else Callback(),
        checkpoint_callback,
        timer,
    ],
        devices=num_gpu,
        detect_anomaly=True,
        max_epochs=num_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_train_batches,
        gradient_clip_val=1000.0 if "imagenet" in dataset_name else 100.0,
        accelerator="gpu",
        logger=logger_)

    trainer.fit(model_corrected, dl_train, dl_val)
    train_time = timer.time_elapsed("train")
    logger.info(f"Training time: {train_time:.2f} s")

    if logger_ is not None:
        logger_.log_metrics({"train_time": train_time, "gpu_name": torch.cuda.get_device_name()})
    
    contains_nans = [n for n, m in model_corrected.named_parameters() if torch.isnan(m).any()]
    assert len(contains_nans) == 0, f"The following params contain NaN values: {contains_nans}"
    
    # Store checkpoint when no finetuning is done
    if config['num_epochs'] == 0 and dataset_name != "imagenet":
        os.makedirs(f"{checkpoint_dir_corrected}/{config_name}", exist_ok=True)
        shutil.copy(ckpt_path, f"{checkpoint_dir_corrected}/{config_name}/last.ckpt")
    logger.info(f"Stored model at {checkpoint_dir_corrected}/{config_name}/last.ckpt")

if __name__ == "__main__":
    main()
