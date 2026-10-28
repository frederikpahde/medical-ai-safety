# Attacked Dataset Generator for Medical AI Safety

This module provides tools for generating "attacked" datasets with artificial artifacts, designed to create controlled spurious correlations for studying and mitigating bias in deep learning models for medical imaging.

## Overview

The tools in this repository allow researchers to:
- Add various types of artificial artifacts to images
- Control which classes receive artifacts (creating spurious correlations)
- Control the proportion of samples that receive artifacts
- Generate ground-truth masks for artifact localization
- Save metadata about the attacked dataset

## Repository Structure

```
├── generate_attacked_dataset.py      # Main script for generating attacked datasets
├── utils/
│   └── artificial_artifact.py        # Utility module with artifact functions
├── config_files/
│   └── config_generator_attacked_dataset.py  # Config file generator
└── README_ATTACKED_DATASET.md        # This file
```

## Supported Artifact Types

| Artifact Type | Description | Common Use Case |
|--------------|-------------|-----------------|
| `timestamp` | Adds date/time text overlay | HyperKvasir endoscopy |
| `microscope` | Circular border/vignette | Dermoscopy images |
| `brightness` | Uniform brightness increase | CheXpert X-rays |
| `lsb` | LSB watermark (invisible) | ISIC controlled |
| `noise` | Static noise pattern | PTB-XL ECG |
| `colored_square` | Colored square in corner | Generic bias |
| `ruler` | Ruler-like edge artifact | ISIC dermoscopy |
| `band_aid` | Band-aid shaped artifact | ISIC dermoscopy |
| `circle` | Circular marker | Equipment markers |
| `text` | Custom text overlay | Generic |
| `blur` | Gaussian blur | Motion artifacts |

## Installation

```bash
pip install Pillow numpy
```

Optional for some features:
```bash
pip install opencv-python pandas pyyaml
```

## Usage

### Command Line

Basic usage:
```bash
python generate_attacked_dataset.py \
    --dataset_name isic \
    --dataset_path ./datasets/ISIC_2019_Training_Input \
    --attacked_classes 0 \
    --artifact_type timestamp \
    --p_artifact 0.9
```

With custom artifact parameters:
```bash
python generate_attacked_dataset.py \
    --dataset_name chexpert \
    --dataset_path ./datasets/chexpert/train \
    --attacked_classes 0 1 \
    --artifact_type brightness \
    --p_artifact 0.8 \
    --artifact_kwargs '{"brightness_factor": 2.0}'
```

### Python API

```python
from generate_attacked_dataset import AttackedDatasetGenerator

# Create generator
generator = AttackedDatasetGenerator(
    dataset_name='isic',
    dataset_path='./datasets/ISIC_2019_Training_Input',
    attacked_classes=[0],  # Melanoma class
    artifact_type='timestamp',
    p_artifact=0.9,
    seed=42
)

# Generate attacked dataset
output_path = generator.generate()
print(f"Attacked dataset saved to: {output_path}")
```

### Using Artifact Functions Directly

```python
from utils.artificial_artifact import (
    add_timestamp,
    add_ruler,
    increase_brightness,
    apply_artifact
)
from PIL import Image

# Load image
image = Image.open('sample.jpg')

# Add timestamp
modified_image, mask = add_timestamp(
    image,
    position='top_left',
    font_size=16
)

# Or use the generic apply function
modified_image, mask = apply_artifact(
    image,
    artifact_type='ruler',
    position='bottom'
)

# Save results
modified_image.save('modified.jpg')
Image.fromarray(mask).save('mask.png')
```

## Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--dataset_name` | str | Yes | Dataset name (isic, chexpert, hyper_kvasir, bone_age, imagenet, generic) |
| `--dataset_path` | str | Yes | Path to the original dataset |
| `--attacked_classes` | int+ | Yes | Class indices to add artifacts to |
| `--artifact_type` | str | Yes | Type of artifact to add |
| `--p_artifact` | float | No | Proportion of samples to attack (default: 0.9) |
| `--output_dir` | str | No | Output directory (default: dataset parent) |
| `--seed` | int | No | Random seed (default: 42) |
| `--artifact_kwargs` | str | No | JSON string of artifact parameters |

## Output Structure

The generated attacked dataset has the following structure:

```
attacked_<artifact_type>_<p_artifact*100>/
├── <original_folder_structure>/
│   └── *.jpg (modified images)
├── masks/
│   └── <original_folder_structure>/
│       └── *.png (artifact masks)
└── artifact_metadata.json
```

### Metadata File

The `artifact_metadata.json` contains:
```json
{
    "dataset_name": "isic",
    "artifact_type": "timestamp",
    "p_artifact": 0.9,
    "attacked_classes": [0],
    "seed": 42,
    "created_at": "2024-01-15T10:30:00",
    "total_images": 25331,
    "images_with_artifact": 2048,
    "samples": [
        {
            "image": "ISIC_0000001.jpg",
            "label": 0,
            "has_artifact": true,
            "mask_path": "masks/ISIC_0000001.png"
        },
        ...
    ]
}
```

## Configuration Files

You can also use YAML configuration files:

```yaml
# config_files/attacked_datasets/isic_timestamp.yaml
dataset:
  name: isic
  path: datasets/ISIC_2019_Training_Input

artifact:
  type: timestamp
  p_artifact: 0.9
  attacked_classes: [0]
  kwargs:
    position: top_left
    font_size: 16

general:
  seed: 42
```

Generate config files:
```bash
python config_files/config_generator_attacked_dataset.py
```

## Dataset-Specific Notes

### ISIC 2019
- Expects `ISIC_2019_Training_GroundTruth.csv` in parent directory
- Classes: MEL(0), NV(1), BCC(2), AK(3), BKL(4), DF(5), VASC(6), SCC(7)
- Common artifacts: ruler, band_aid, skin_marker, timestamp, lsb

### HyperKvasir
- Uses folder-based class structure
- Common artifacts: timestamp, insertion_tube (microscope)

### CheXpert
- Common artifacts: brightness (simulating exposure variations)
- Class based on pathology labels

### PTB-XL (ECG)
- Uses `add_ecg_static_noise()` for 1D signal artifacts
- Artifacts applied to specific leads

## Integration with medical-ai-safety Repository

To integrate with the main repository:

1. Copy `generate_attacked_dataset.py` to root
2. Copy `utils/artificial_artifact.py` to `utils/`
3. Copy `config_files/config_generator_attacked_dataset.py` to `config_files/`

Then use in training:
```bash
# Generate attacked dataset
python generate_attacked_dataset.py \
    --dataset_name isic \
    --dataset_path ./datasets/ISIC_2019_Training_Input \
    --attacked_classes 0 \
    --artifact_type timestamp \
    --p_artifact 0.9

# Train with attacked dataset
python -m model_training.start_training \
    --config_file config_files/training/isic_attacked/config.yaml
```

## Examples

### Creating Multiple Attacked Versions

```python
from generate_attacked_dataset import AttackedDatasetGenerator

artifact_types = ['timestamp', 'ruler', 'brightness', 'lsb']
p_values = [0.5, 0.7, 0.9]

for artifact in artifact_types:
    for p in p_values:
        generator = AttackedDatasetGenerator(
            dataset_name='isic',
            dataset_path='./datasets/ISIC_2019_Training_Input',
            attacked_classes=[0],
            artifact_type=artifact,
            p_artifact=p,
        )
        generator.generate()
```

### Custom Artifact Parameters

```python
# Timestamp with specific formatting
generator = AttackedDatasetGenerator(
    dataset_name='hyper_kvasir',
    dataset_path='./datasets/hyper-kvasir',
    attacked_classes=[0, 1],
    artifact_type='timestamp',
    p_artifact=0.9,
    position='bottom_right',
    font_size=20,
    text_color=(0, 255, 0),
    bg_color=(0, 0, 0),
)

# Brightness with specific factor
generator = AttackedDatasetGenerator(
    dataset_name='chexpert',
    dataset_path='./datasets/chexpert',
    attacked_classes=[0],
    artifact_type='brightness',
    p_artifact=0.9,
    brightness_factor=2.0,
)
```

## References

This implementation is based on the methodology described in:

- Pahde et al., "Ensuring Medical AI Safety: Explainable AI-Driven Detection and Mitigation of Spurious Model Behavior and Associated Data" (2024)
- Repository: https://github.com/frederikpahde/medical-ai-safety

## License

This code is provided for research purposes. Please cite the original paper if you use this in your research.
