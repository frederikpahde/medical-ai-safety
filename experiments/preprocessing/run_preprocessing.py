import os
from argparse import ArgumentParser

import yaml
from datasets import DATASET_CLASSES

from experiments.preprocessing.global_collect_relevances_and_activations import run_collect_relevances_and_activations


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--split', default="all")
    parser.add_argument("--force_recompute", default=False, type=bool)
    parser.add_argument('--config_file', default=None)        
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    config['config_file'] = args.config_file
    run_preprocessing(config, args.split, args.force_recompute)


def run_preprocessing(config, split, force_recompute):

    classes = DATASET_CLASSES[config["dataset_name"]].classes
    for class_idx in range(len(classes)):
        run_collect_relevances_and_activations({**config,
                                                'class_idx': class_idx,
                                                'split': split},
                                                force_recompute)

if __name__ == "__main__":
    main()
