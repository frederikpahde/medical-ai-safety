from argparse import ArgumentParser

from utils.dora.dora import Dora
from utils.dora.objectives import ChannelObjective
from utils.dora.model import get_dim, modify_model

from datasets import DATASET_CLASSES, DATASET_NORM_PARAMS
from models import get_fn_model_loader
from utils.helper import load_config
import os
import torch
import torchvision.transforms as transforms


def get_parser():
    parser = ArgumentParser(
        description='Generate sAMS for DORA Analysis.', )

    parser.add_argument('--n', default=5, type=int)
    parser.add_argument('--k', default=None, type=int)
    parser.add_argument('--iters', default=1, type=int)
    parser.add_argument('--aggr', default="max", type=str)
    parser.add_argument('--config_file',
                        default="config_files/revealing/isic/local/resnet50d_identity_2.yaml", )
   
    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    run_dora_preprocessing(config, args.n, args.k, args.aggr, args.iters)


def run_dora_preprocessing(config, n, k, aggr, iters):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(DATASET_CLASSES[config["dataset_name"]].classes)

    model_name = config["model_name"]
    
    model = get_fn_model_loader(model_name)(n_class=n_classes,
                                            ckpt_path=config["ckpt_path"]
                                            ).to(device).eval()

    savedir = f"{config['dir_precomputed_data']}/dora_data/{config['dataset_name']}_{config['model_name']}_{aggr}"
    os.makedirs(savedir, exist_ok=True)

    experiment_name = config["config_name"]

    model = modify_model(model, config["layer_name"], aggr=aggr)
    k = get_dim(model, config["img_size"], device) if k is None else k
    neuron_indices = [i for i in range(0, k)]

    mean, std = DATASET_NORM_PARAMS[config['dataset_name']]
    fn_normalize = transforms.Normalize(mean=mean, std=std)

    print(f"Using mean: {mean}, std: {std}")
    d = Dora(model=model,
            storage_dir=savedir,
            device=device)
    
    d.generate_signals(
        neuron_idx=neuron_indices,
        num_samples = n,
        layer=model.pool,
        only_maximization = True,
        fn_normalize=fn_normalize,
        image_transforms = transforms.Compose([transforms.Pad(2, fill=.5, padding_mode='constant'),
                                                transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
                                                transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
                                                transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
                                                transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
                                                transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
                                                transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
                                                transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
                                                transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
                                                transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
                                                transforms.RandomAffine(0, translate=(0.015, 0.015), fill=0.5),
                                                transforms.RandomAffine((-20,20),
                                                                        scale=(0.75, 1.025),
                                                                        fill=0.5),
                                                transforms.RandomCrop((224, 224),
                                                                    padding=None,
                                                                    pad_if_needed=True,
                                                                    fill=0,
                                                                    padding_mode='constant')]),
        objective_fn=ChannelObjective(),
        lr=0.05,
        width=224,
        height=224,
        iters=iters,
        batch_size=16,
        experiment_name=experiment_name,
        overwrite_experiment=True, 
    )

    
if __name__ == "__main__":
    main()
