import copy
import logging
import os
from argparse import ArgumentParser

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import load_dataset
from model_training.train_model import train_model
from utils.training_utils import get_optimizer, get_loss
from models import get_fn_model_loader
from utils.plots import visualize_dataset

torch.multiprocessing.set_sharing_strategy('file_system')


def get_parser():
    parser = ArgumentParser(description='Train models.', )
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--visualize_datasets', type=bool, default=None)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    config_name = os.path.basename(config_file)[:-5]
    start_training(config, config_name, args.visualize_datasets)


def start_training(config, config_name, visualize_datasets):
    """ Starts training for given config file.

    Args:
        config (dict): Dictionary with config parameters for training.
        config_name (str): Name of given config
    """

    dataset_name = config['dataset_name']
    model_name = config['model_name']
    pretrained = config.get('pretrained', False)
    num_epochs = config['num_epochs']
    eval_every_n_epochs = config['eval_every_n_epochs']
    store_every_n_epochs = config['store_every_n_epochs']
    batch_size = config['batch_size']
    optimizer_name = config['optimizer']
    clean_samples_only = config.get('clean_samples_only', False)
    ckpt_path = config.get('ckpt_path', None)
    start_epoch = torch.load(ckpt_path)["epoch"] + 1 if ckpt_path is not None else 0
    compute_per_class_metrics = config.get("compute_per_class_metrics", False)
    loss_name = config['loss']
    lr = config['lr']
    model_savedir = config['model_savedir']
    percentage_batches = config.get('percentage_batches', 1)

    # Attack Details
    attacked_classes = config.get('attacked_classes', [])

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config.get('device', default_device)
    wandb_project_name = config.get('wandb_project_name', None)
    wandb_api_key = config.get('wandb_api_key', None)

    do_wandb_logging = wandb_project_name is not None

    # Initialize WandB
    if do_wandb_logging:
        assert wandb_api_key is not None, f"'wandb_api_key' required if 'wandb_project_name' is provided ({wandb_project_name})"
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.init(project=wandb_project_name, config=config)
        wandb.run.name = f"{config_name}-{wandb.run.name}"
        logger.info(f"Initialized wand. Logging to {wandb_project_name} / {wandb.run.name}...")

    dataset= load_dataset(config)

    fn_model_loader = get_fn_model_loader(model_name)

    num_classes = len(dataset.classes)

    logger.info(f"Loading model with ckpt_path {ckpt_path}")
    model = fn_model_loader(
        ckpt_path=ckpt_path,
        pretrained=pretrained,
        n_class=num_classes,
        device=device).to(device)

    # Define Optimizer and Loss function
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, ckpt_path)
    criterion = get_loss(loss_name, weights=dataset.weights)

    idxs_train = dataset.idxs_train if percentage_batches is None else dataset.idxs_train[::int(1 / percentage_batches)]
    idxs_val = dataset.idxs_val if percentage_batches is None else dataset.idxs_val[::int(1 / percentage_batches)]
    dataset_train = dataset.get_subset_by_idxs(idxs_train)
    dataset_val = dataset.get_subset_by_idxs(idxs_val)
    dataset_test = dataset.get_subset_by_idxs(dataset.idxs_test)

    logger.info(
        f"Splitting the data into train ({len(dataset_train)}) and val ({len(dataset_val)}), ignoring samples from test ({len(dataset.idxs_test)})")

    dataset_train.do_augmentation = True
    dataset_val.do_augmentation = False
    dataset_test.do_augmentation = False

    if clean_samples_only:
        logger.info(f"#Samples before filtering: {len(dataset_train)}")
        dataset_train = dataset_train.get_subset_by_idxs(dataset_train.clean_sample_ids)
        logger.info(f"#Samples after filtering: {len(dataset_train)}")

    logger.info(f"Number of samples: {len(dataset_train)} (train) / {len(dataset_val)} (val)")

    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    dl_val_dict = {'val': DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)}

    if (len(attacked_classes) > 0):

        config_clean = copy.deepcopy(config)
        config_clean["p_artifact"] = 0.0
        dataset_clean= load_dataset(config_clean)

        if "imagenet" in dataset_name:
            all_classes = list(dataset.label_map.keys())
        elif "bone" in dataset_name:
            all_classes = range(len(dataset.classes))
        elif "chexpert" in dataset_name:
            all_classes = range(len(dataset.classes))
        elif "ptb" in dataset_name:
            all_classes = range(len(dataset.classes))
        else:
            all_classes = dataset.classes 

        config_attacked = copy.deepcopy(config)
        config_attacked["attacked_classes"] = all_classes
        config_attacked["p_artifact"] = 1.0
        dataset_attacked = load_dataset(config_attacked)

        dataset_val_clean = dataset_clean.get_subset_by_idxs(dataset.idxs_val)
        dataset_test_clean = dataset_clean.get_subset_by_idxs(dataset.idxs_test)
        dataset_val_attacked = dataset_attacked.get_subset_by_idxs(dataset.idxs_val)
        dataset_test_attacked = dataset_attacked.get_subset_by_idxs(dataset.idxs_test)
        
        dl_val_dict['val_clean'] = DataLoader(dataset_val_clean, batch_size=batch_size, shuffle=False, num_workers=8)
        dl_val_dict['test_clean'] = DataLoader(dataset_test_clean, batch_size=batch_size, shuffle=False, num_workers=8)
        dl_val_dict['val_attacked'] = DataLoader(dataset_val_attacked, batch_size=batch_size, shuffle=False,
                                                 num_workers=8)
        dl_val_dict['test_attacked'] = DataLoader(dataset_test_attacked, batch_size=batch_size, shuffle=False,
                                                 num_workers=8)
            
    milestones = [int(x) for x in config.get("milestones", "30,40").split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1, last_epoch=-1)
    
    # Set scheduler to correct step
    for _ in range(start_epoch):
        scheduler.step()

    visualization_path = f"datasets_visualized/{dataset_name}"
    
    if visualize_datasets:
        os.makedirs(visualization_path, exist_ok=True)
        start_idx = max(0, dataset_val.artifact_ids[0] - 10) if hasattr(dataset_val, "artifact_ids") else 0
        fname = f"dataset_attacked{attacked_classes[0]}_normal.png" if len(attacked_classes) > 0 else f"dataset_normal.png"
        visualize_dataset(dataset_val, f"{visualization_path}/{fname}", start_idx)
        fname = f"dataset_attacked{attacked_classes[0]}_clean.png" if len(attacked_classes) > 0 else f"dataset_clean.png"
        visualize_dataset(dl_val_dict['test_clean'].dataset, f"{visualization_path}/{fname}", start_idx)
        fname = f"dataset_attacked{attacked_classes[0]}_attacked.png" if len(attacked_classes) > 0 else f"dataset_attacked.png"
        visualize_dataset(dl_val_dict['test_attacked'].dataset, f"{visualization_path}/{fname}", start_idx)
        logger.info("Visualized datasets")

    # Start Training
    train_model(
        model,
        model_name,
        dl_train,
        dl_val_dict,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        eval_every_n_epochs,
        store_every_n_epochs,
        device,
        f"{model_savedir}/{config_name}",
        do_wandb_logging,
        start_epoch,
        percentage_batches=1,
        compute_per_class_metrics=compute_per_class_metrics
    )

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
