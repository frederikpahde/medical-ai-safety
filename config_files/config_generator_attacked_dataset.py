"""
Config File Generator for Attacked Dataset Generation
=====================================================
This script generates configuration files for creating attacked datasets
with various artificial artifacts.

This file should be placed at: config_files/config_generator_attacked_dataset.py

Usage:
------
python config_files/config_generator_attacked_dataset.py

This will generate config files in config_files/attacked_datasets/
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any


def generate_config(
    dataset_name: str,
    dataset_path: str,
    attacked_classes: List[int],
    artifact_type: str,
    p_artifact: float = 0.9,
    seed: int = 42,
    output_dir: str = None,
    artifact_kwargs: Dict[str, Any] = None
) -> Dict:
    """
    Generate a configuration dictionary for attacked dataset creation.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the original dataset
        attacked_classes: List of class indices to attack
        artifact_type: Type of artifact to add
        p_artifact: Proportion of samples to attack
        seed: Random seed
        output_dir: Output directory for attacked dataset
        artifact_kwargs: Additional artifact parameters
        
    Returns:
        Configuration dictionary
    """
    config = {
        'dataset': {
            'name': dataset_name,
            'path': dataset_path,
        },
        'artifact': {
            'type': artifact_type,
            'p_artifact': p_artifact,
            'attacked_classes': attacked_classes,
        },
        'general': {
            'seed': seed,
        }
    }
    
    if output_dir:
        config['output'] = {'dir': output_dir}
    
    if artifact_kwargs:
        config['artifact']['kwargs'] = artifact_kwargs
    
    return config


def save_config(config: Dict, filepath: str):
    """Save configuration to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def generate_all_configs():
    """Generate all configuration files for attacked datasets."""
    
    # Create output directory
    config_dir = Path('config_files/attacked_datasets')
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # ISIC 2019 Configurations
    # ==========================================================================
    
    isic_configs = [
        {
            'name': 'isic_timestamp_class0',
            'config': generate_config(
                dataset_name='isic',
                dataset_path='datasets/ISIC_2019_Training_Input',
                attacked_classes=[0],  # Melanoma
                artifact_type='timestamp',
                p_artifact=0.9,
            )
        },
        {
            'name': 'isic_lsb_class0',
            'config': generate_config(
                dataset_name='isic',
                dataset_path='datasets/ISIC_2019_Training_Input',
                attacked_classes=[0],
                artifact_type='lsb',
                p_artifact=0.9,
                artifact_kwargs={'pattern_size': 8}
            )
        },
        {
            'name': 'isic_ruler_class0',
            'config': generate_config(
                dataset_name='isic',
                dataset_path='datasets/ISIC_2019_Training_Input',
                attacked_classes=[0],
                artifact_type='ruler',
                p_artifact=0.9,
            )
        },
        {
            'name': 'isic_band_aid_class0',
            'config': generate_config(
                dataset_name='isic',
                dataset_path='datasets/ISIC_2019_Training_Input',
                attacked_classes=[0],
                artifact_type='band_aid',
                p_artifact=0.9,
            )
        },
        {
            'name': 'isic_microscope_class0',
            'config': generate_config(
                dataset_name='isic',
                dataset_path='datasets/ISIC_2019_Training_Input',
                attacked_classes=[0],
                artifact_type='microscope',
                p_artifact=0.9,
            )
        },
    ]
    
    # ==========================================================================
    # HyperKvasir Configurations
    # ==========================================================================
    
    hyper_kvasir_configs = [
        {
            'name': 'hyper_kvasir_timestamp',
            'config': generate_config(
                dataset_name='hyper_kvasir',
                dataset_path='datasets/hyper-kvasir-labeled-images/labeled-images',
                attacked_classes=[0, 1, 2],  # First 3 classes
                artifact_type='timestamp',
                p_artifact=0.9,
            )
        },
        {
            'name': 'hyper_kvasir_circle',
            'config': generate_config(
                dataset_name='hyper_kvasir',
                dataset_path='datasets/hyper-kvasir-labeled-images/labeled-images',
                attacked_classes=[0],
                artifact_type='microscope',
                p_artifact=0.9,
            )
        },
    ]
    
    # ==========================================================================
    # CheXpert Configurations
    # ==========================================================================
    
    chexpert_configs = [
        {
            'name': 'chexpert_brightness_class0',
            'config': generate_config(
                dataset_name='chexpert',
                dataset_path='datasets/chexpert/train',
                attacked_classes=[0],
                artifact_type='brightness',
                p_artifact=0.9,
                artifact_kwargs={'brightness_factor': 1.5}
            )
        },
        {
            'name': 'chexpert_brightness_high_class0',
            'config': generate_config(
                dataset_name='chexpert',
                dataset_path='datasets/chexpert/train',
                attacked_classes=[0],
                artifact_type='brightness',
                p_artifact=0.9,
                artifact_kwargs={'brightness_factor': 2.0}
            )
        },
    ]
    
    # ==========================================================================
    # Bone Age Configurations
    # ==========================================================================
    
    bone_age_configs = [
        {
            'name': 'bone_age_brightness_class0',
            'config': generate_config(
                dataset_name='bone_age',
                dataset_path='datasets/boneage-training-dataset',
                attacked_classes=[0],
                artifact_type='brightness',
                p_artifact=0.9,
                artifact_kwargs={'brightness_factor': 1.5}
            )
        },
        {
            'name': 'bone_age_colored_square_class0',
            'config': generate_config(
                dataset_name='bone_age',
                dataset_path='datasets/boneage-training-dataset',
                attacked_classes=[0],
                artifact_type='colored_square',
                p_artifact=0.9,
                artifact_kwargs={'color': (255, 255, 255), 'corner': 'bottom_right'}
            )
        },
    ]
    
    # ==========================================================================
    # ImageNet Configurations
    # ==========================================================================
    
    imagenet_configs = [
        {
            'name': 'imagenet_timestamp_class0',
            'config': generate_config(
                dataset_name='imagenet',
                dataset_path='datasets/imagenet/train',
                attacked_classes=[0],
                artifact_type='timestamp',
                p_artifact=0.9,
            )
        },
    ]
    
    # ==========================================================================
    # Generic/Custom Configurations (for user's own datasets)
    # ==========================================================================
    
    generic_configs = [
        {
            'name': 'generic_timestamp_example',
            'config': generate_config(
                dataset_name='generic',
                dataset_path='datasets/your_dataset',
                attacked_classes=[0],
                artifact_type='timestamp',
                p_artifact=0.9,
            )
        },
        {
            'name': 'generic_all_artifacts_example',
            'config': {
                'dataset': {
                    'name': 'generic',
                    'path': 'datasets/your_dataset',
                },
                'artifact': {
                    'type': 'timestamp',  # Change this
                    'p_artifact': 0.9,
                    'attacked_classes': [0],
                    'available_types': [
                        'timestamp',
                        'microscope', 
                        'brightness',
                        'lsb',
                        'noise',
                        'colored_square',
                        'ruler',
                        'band_aid',
                    ]
                },
                'general': {
                    'seed': 42,
                },
                'notes': 'This is a template config. Modify artifact.type and dataset.path for your use case.'
            }
        },
    ]
    
    # Combine all configs
    all_configs = (
        isic_configs + 
        hyper_kvasir_configs + 
        chexpert_configs + 
        bone_age_configs + 
        imagenet_configs +
        generic_configs
    )
    
    # Save all configs
    print(f"Generating {len(all_configs)} configuration files...")
    
    for cfg in all_configs:
        filepath = config_dir / f"{cfg['name']}.yaml"
        save_config(cfg['config'], filepath)
        print(f"  Created: {filepath}")
    
    print(f"\nDone! Configuration files saved to {config_dir}/")


if __name__ == '__main__':
    generate_all_configs()
