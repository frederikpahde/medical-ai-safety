from argparse import ArgumentParser
import os
import torch
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.visualization import FeatureVisualization
from zennit.composites import EpsilonPlusFlat, EpsilonPlus

from datasets import load_dataset
from models import get_fn_model_loader, get_canonizer
from utils.helper import get_layer_names_with_identites, load_config, none_or_int


def get_parser():
    parser = ArgumentParser(
        description='Run CRP preprocessing.', )

    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--class_id', default=None, type=none_or_int)
    parser.add_argument('--config_file',
                        default="config_files/revealing/isic/local/resnet50d_identity_2.yaml")
    return parser


def main():
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    run_crp(config, args.split, args.class_id)

def run_crp(config, split, class_id):
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fv_name = f"{config['dir_precomputed_data']}/crp_files/{dataset_name}_{split}_{config['model_name']}"
    if os.path.isdir(f"{fv_name}/ActMax_sum_normed"):
        print(f"CRP files already exist for {config['config_name']} at {fv_name}")
        return
    
    dataset = load_dataset(config, normalize_data=False)

    splits = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
        }

    dataset = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])
    
    if "ptb" in dataset_name:
        dataset.make_single_label()

    

    if class_id is not None:
        fv_name = f"{fv_name}_class{class_id}"
        idxs_class = [i for i in range(len(dataset)) if dataset.get_target(i) == class_id]
        dataset = dataset.get_subset_by_idxs(idxs_class)

    print(f"Run CRP preprocessing with {len(dataset)} samples.")

    model = get_fn_model_loader(model_name)(n_class=len(dataset.classes),
                                            ckpt_path=config["ckpt_path"],
                                            device=device
                                            ).to(device).eval()

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlus(canonizers) if "ptb" in dataset_name else EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_names = get_layer_names_with_identites(model)

    print(f"Storing acts/rels for {layer_names}")
    layer_map = {layer: cc for layer in layer_names}

    attribution = CondAttribution(model)

    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.normalize_fn,
                              path=fv_name, cache=None)

    batch_size = config["batch_size"]
    while len(dataset) % batch_size == 1:
        batch_size += 1
    print(f"Using batch_size {batch_size}")
    fv.run(composite, 0, len(dataset), batch_size=batch_size)

if __name__ == "__main__":
    main()
