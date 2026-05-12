#!/usr/bin/env python3
"""
Generate Attacked Dataset Script
================================
This script generates a dataset with artificial artifacts added to a specified
proportion of samples from designated classes OR samples containing a specific
existing artifact, creating controlled spurious correlations for medical AI 
safety research.

The script is designed to be compatible with the medical-ai-safety repository:
https://github.com/frederikpahde/medical-ai-safety

Features:
---------
1. Class-based attack: Add artifacts to samples of specific classes
2. Artifact-to-artifact attack: Add artifacts to samples that already contain 
   a specific existing artifact (irrespective of class)

Usage:
------
# Class-based attack:
python generate_attacked_dataset.py \
    --dataset_name isic \
    --dataset_path /ritter/roshan/workspace/ICON/experiments/melanoma/dataset/ \
    --attacked_classes 0 \
    --artifact_type timestamp \
    --p_artifact 0.9

# Artifact-to-artifact attack:
python generate_attacked_dataset.py \
    --dataset_name isic \
    --dataset_path /ritter/roshan/workspace/ICON/experiments/melanoma/dataset/microscope_40/ \
    --attacked_artifact microscope \
    --artifact_type timestamp \
    --p_artifact_to_artifact 0.4

Supported Artifact Types:
-------------------------
- timestamp: Adds a timestamp text overlay (commonly used in HyperKvasir)
- microscope: Adds a black circular border (simulating microscope view)
- brightness: Increases image brightness (commonly used in CheXpert)
- lsb: LSB watermark attack (commonly used in ISIC)
- noise: Adds static noise pattern (commonly used in PTB-XL for ECG)
- colored_square: Adds a colored square to corner
- ruler: Adds a ruler-like artifact
- band_aid: Adds a band-aid-like artifact

Author: Generated for medical-ai-safety repository
"""

import argparse
import os
import shutil
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")


# =============================================================================
# Artifact Generation Functions
# =============================================================================

class ArtifactGenerator:
    """
    Class to generate various types of artificial artifacts for medical images.
    
    These artifacts are used to create controlled spurious correlations in 
    datasets for studying and mitigating bias in deep learning models.
    """
    
    def __init__(self, artifact_type: str, seed: int = 42, **kwargs):
        """
        Initialize the artifact generator.
        
        Args:
            artifact_type: Type of artifact to generate
            seed: Random seed for reproducibility
            **kwargs: Additional parameters for specific artifact types
        """
        self.artifact_type = artifact_type
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.kwargs = kwargs
        
        # Map artifact types to generation functions
        self.artifact_functions = {
            'timestamp': self.add_timestamp,
            'microscope': self.add_microscope_border,
            'brightness': self.increase_brightness,
            'lsb': self.add_lsb_watermark,
            'noise': self.add_static_noise,
            'colored_square': self.add_colored_square,
            'ruler': self.add_ruler,
            'band_aid': self.add_band_aid,
            'circle': self.add_circle_artifact,
            'text': self.add_text_overlay,
            'blur': self.add_blur_artifact,
        }
        
        if artifact_type not in self.artifact_functions:
            raise ValueError(f"Unknown artifact type: {artifact_type}. "
                           f"Available types: {list(self.artifact_functions.keys())}")
    
    def apply(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Union[Image.Image, np.ndarray], np.ndarray]:
        """
        Apply the artifact to an image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Tuple of (modified_image, artifact_mask)
        """
        return self.artifact_functions[self.artifact_type](image)
    
    def add_timestamp(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Add a timestamp text overlay to the image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.copy()
        width, height = image.size
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Generate timestamp text
        timestamp_text = self.kwargs.get('timestamp_text', None)
        if timestamp_text is None:
            year = self.rng.randint(2015, 2024)
            month = self.rng.randint(1, 13)
            day = self.rng.randint(1, 29)
            hour = self.rng.randint(0, 24)
            minute = self.rng.randint(0, 60)
            timestamp_text = f"{year}/{month:02d}/{day:02d} {hour:02d}:{minute:02d}"
        
        draw = ImageDraw.Draw(image)
        font_size = self.kwargs.get('font_size', max(24, min(width, height) // 14))

        _font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",           # Debian/Ubuntu
            "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",                # RHEL/CentOS/HPC
            "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
            "/usr/share/fonts/google-droid/DroidSans.ttf",
        ]
        font = ImageFont.load_default()
        for _p in _font_paths:
            try:
                font = ImageFont.truetype(_p, font_size)
                break
            except (IOError, OSError):
                continue
        
        position = self.kwargs.get('position', 'top_left')
        padding = self.kwargs.get('padding', 10)
        
        bbox = draw.textbbox((0, 0), timestamp_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        if position == 'top_left':
            x, y = padding, padding
        elif position == 'top_right':
            x, y = width - text_width - padding, padding
        elif position == 'bottom_left':
            x, y = padding, height - text_height - padding
        elif position == 'bottom_right':
            x, y = width - text_width - padding, height - text_height - padding
        else:
            x, y = padding, padding
        
        bg_color = self.kwargs.get('bg_color', None)
        text_color = self.kwargs.get('text_color', (255, 255, 0))

        if bg_color is not None:
            draw.rectangle([x - 2, y - 2, x + text_width + 2, y + text_height + 2], fill=bg_color)
        draw.text((x, y), timestamp_text, font=font, fill=text_color)
        
        mask[y-2:y+text_height+2, x-2:x+text_width+2] = 255
        
        return image, mask
    
    def add_microscope_border(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Add a black circular border simulating a microscope/dermoscope view.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.copy()
        width, height = image.size
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        center_x, center_y = width // 2, height // 2
        radius = self.kwargs.get('radius', min(width, height) // 2 - 10)
        
        img_array = np.array(image)
        
        for x in range(width):
            for y in range(height):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist > radius:
                    img_array[y, x] = (0, 0, 0) if len(img_array.shape) == 3 else 0
                    mask[y, x] = 255
        
        return Image.fromarray(img_array), mask
    
    def increase_brightness(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Increase image brightness uniformly.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        width, height = image.size
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        brightness_factor = self.kwargs.get('brightness_factor', 1.5)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        return image, mask
    
    def add_lsb_watermark(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Add LSB (Least Significant Bit) watermark to the image.
        """
        if isinstance(image, np.ndarray):
            img_array = image.copy()
        else:
            img_array = np.array(image)
        
        height, width = img_array.shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        pattern_size = self.kwargs.get('pattern_size', 8)
        pattern = self.rng.randint(0, 2, (pattern_size, pattern_size)).astype(np.uint8)
        
        full_pattern = np.tile(pattern, (height // pattern_size + 1, width // pattern_size + 1))
        full_pattern = full_pattern[:height, :width]
        
        if len(img_array.shape) == 3:
            for c in range(img_array.shape[2]):
                img_array[:, :, c] = (img_array[:, :, c] & 0xFE) | full_pattern
        else:
            img_array = (img_array & 0xFE) | full_pattern
        
        if isinstance(image, Image.Image):
            return Image.fromarray(img_array), mask
        return img_array, mask
    
    def add_static_noise(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Add static noise pattern to the image.
        """
        if isinstance(image, np.ndarray):
            img_array = image.copy().astype(np.float32)
        else:
            img_array = np.array(image).astype(np.float32)
        
        height, width = img_array.shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        noise_level = self.kwargs.get('noise_level', 25)
        noise = self.rng.randn(*img_array.shape) * noise_level
        
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        if isinstance(image, Image.Image):
            return Image.fromarray(img_array), mask
        return img_array, mask
    
    def add_colored_square(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Add a colored square to a corner of the image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.copy()
        width, height = image.size
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        square_size = self.kwargs.get('square_size', min(width, height) // 10)
        color = self.kwargs.get('color', (255, 0, 0))
        corner = self.kwargs.get('corner', 'bottom_right')
        
        if corner == 'top_left':
            x1, y1 = 0, 0
        elif corner == 'top_right':
            x1, y1 = width - square_size, 0
        elif corner == 'bottom_left':
            x1, y1 = 0, height - square_size
        else:
            x1, y1 = width - square_size, height - square_size
        
        x2, y2 = x1 + square_size, y1 + square_size
        
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], fill=color)
        
        mask[y1:y2, x1:x2] = 255
        
        return image, mask
    
    def add_ruler(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Add a ruler-like artifact to the image edge.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.copy()
        width, height = image.size
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        draw = ImageDraw.Draw(image)
        
        ruler_width = self.kwargs.get('ruler_width', max(10, width // 20))
        ruler_color = self.kwargs.get('ruler_color', (0, 0, 0))
        tick_color = self.kwargs.get('tick_color', (255, 255, 255))
        position = self.kwargs.get('position', 'bottom')
        
        if position == 'bottom':
            draw.rectangle([0, height - ruler_width, width, height], fill=ruler_color)
            mask[height - ruler_width:height, :] = 255
            
            tick_spacing = width // 20
            for i in range(0, width, tick_spacing):
                tick_height = ruler_width // 2 if i % (tick_spacing * 2) == 0 else ruler_width // 4
                draw.line([(i, height - tick_height), (i, height)], fill=tick_color, width=2)
        
        elif position == 'right':
            draw.rectangle([width - ruler_width, 0, width, height], fill=ruler_color)
            mask[:, width - ruler_width:width] = 255
            
            tick_spacing = height // 20
            for i in range(0, height, tick_spacing):
                tick_width = ruler_width // 2 if i % (tick_spacing * 2) == 0 else ruler_width // 4
                draw.line([(width - tick_width, i), (width, i)], fill=tick_color, width=2)
        
        return image, mask
    
    def add_band_aid(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Add a band-aid-like artifact to the image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.copy()
        width, height = image.size
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        band_length = self.kwargs.get('band_length', min(width, height) // 3)
        band_width = self.kwargs.get('band_width', band_length // 6)
        band_color = self.kwargs.get('band_color', (210, 180, 140))
        
        center_x = self.kwargs.get('center_x', self.rng.randint(band_length//2, width - band_length//2))
        center_y = self.kwargs.get('center_y', self.rng.randint(band_width, height - band_width))
        angle = self.kwargs.get('angle', self.rng.uniform(-30, 30))
        
        from PIL import Image as PILImage
        
        band_aid = PILImage.new('RGBA', (band_length, band_width), (0, 0, 0, 0))
        band_draw = ImageDraw.Draw(band_aid)
        band_draw.rounded_rectangle([0, 0, band_length-1, band_width-1], 
                                    radius=band_width//2, fill=band_color + (255,))
        
        for i in range(5, band_length - 5, 8):
            for j in range(2, band_width - 2, 4):
                band_draw.ellipse([i-1, j-1, i+1, j+1], fill=(190, 160, 120, 255))
        
        band_aid = band_aid.rotate(angle, expand=True, resample=Image.BICUBIC)
        
        paste_x = center_x - band_aid.width // 2
        paste_y = center_y - band_aid.height // 2
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        image.paste(band_aid, (paste_x, paste_y), band_aid)
        image = image.convert('RGB')
        
        mask_band = np.array(band_aid.split()[-1])
        mask_band = (mask_band > 0).astype(np.uint8) * 255
        
        py1, py2 = max(0, paste_y), min(height, paste_y + band_aid.height)
        px1, px2 = max(0, paste_x), min(width, paste_x + band_aid.width)
        my1, my2 = max(0, -paste_y), min(band_aid.height, height - paste_y)
        mx1, mx2 = max(0, -paste_x), min(band_aid.width, width - paste_x)
        
        mask[py1:py2, px1:px2] = mask_band[my1:my2, mx1:mx2]
        
        return image, mask
    
    def add_circle_artifact(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Add a circular artifact.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.copy()
        width, height = image.size
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        draw = ImageDraw.Draw(image)
        
        radius = self.kwargs.get('radius', min(width, height) // 15)
        color = self.kwargs.get('color', (255, 255, 255))
        center_x = self.kwargs.get('center_x', self.rng.randint(radius, width - radius))
        center_y = self.kwargs.get('center_y', self.rng.randint(radius, height - radius))
        filled = self.kwargs.get('filled', True)
        
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        
        if filled:
            draw.ellipse(bbox, fill=color)
        else:
            draw.ellipse(bbox, outline=color, width=max(2, radius // 5))
        
        for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
            for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
                if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                    mask[y, x] = 255
        
        return image, mask
    
    def add_text_overlay(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Add custom text overlay to the image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.copy()
        width, height = image.size
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        draw = ImageDraw.Draw(image)
        
        text = self.kwargs.get('text', 'ARTIFACT')
        font_size = self.kwargs.get('font_size', max(12, min(width, height) // 15))
        color = self.kwargs.get('color', (255, 0, 0))
        position = self.kwargs.get('position', (10, 10))
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
        
        bbox = draw.textbbox(position, text, font=font)
        draw.text(position, text, font=font, fill=color)
        
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
        
        return image, mask
    
    def add_blur_artifact(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
        """
        Add blur artifact to a region of the image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        width, height = image.size
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        blur_radius = self.kwargs.get('blur_radius', 5)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return image, mask


# =============================================================================
# Dataset Processing
# =============================================================================

class AttackedDatasetGenerator:
    """
    Generate attacked datasets by adding artifacts to specified classes
    OR to samples containing a specific existing artifact.
    """
    
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        artifact_type: str,
        attacked_classes: Optional[List[int]] = None,
        p_artifact: float = 0.9,
        attacked_artifact: Optional[str] = None,
        p_artifact_to_artifact: Optional[float] = None,
        seed: int = 42,
        **artifact_kwargs
    ):
        """
        Initialize the attacked dataset generator.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'isic', 'chexpert', 'hyper_kvasir')
            dataset_path: Path to the original dataset
            artifact_type: Type of artifact to add
            attacked_classes: List of class indices to add artifacts to (for class-based attack)
            p_artifact: Proportion of attacked class samples to add artifact to
            attacked_artifact: Name of existing artifact to target (for artifact-to-artifact attack)
            p_artifact_to_artifact: Proportion of samples with existing artifact to add new artifact to
            seed: Random seed for reproducibility
            **artifact_kwargs: Additional parameters for artifact generation
        """
        # Validate inputs: cannot specify both attacked_classes and attacked_artifact
        assert not (attacked_classes is not None and attacked_artifact is not None), \
            "Cannot specify both 'attacked_classes' and 'attacked_artifact'. Choose one attack mode."
        
        assert attacked_classes is not None or attacked_artifact is not None, \
            "Must specify either 'attacked_classes' or 'attacked_artifact'."
        
        if attacked_artifact is not None:
            assert p_artifact_to_artifact is not None, \
                "When using 'attacked_artifact', must also specify 'p_artifact_to_artifact'."
        
        self.dataset_name = dataset_name.lower()
        self.dataset_path = Path(dataset_path)
        self.artifact_type = artifact_type
        self.attacked_classes = attacked_classes if attacked_classes is None else \
                               (attacked_classes if isinstance(attacked_classes, list) else [int(attacked_classes)])
        self.p_artifact = p_artifact
        self.attacked_artifact = attacked_artifact
        self.p_artifact_to_artifact = p_artifact_to_artifact
        self.seed = seed
        self.artifact_kwargs = artifact_kwargs
        
        # Determine attack mode
        self.attack_mode = 'class' if attacked_classes is not None else 'artifact'
        
        # Initialize random generator
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        
        # Initialize artifact generator
        self.artifact_generator = ArtifactGenerator(
            artifact_type=artifact_type,
            seed=seed,
            **artifact_kwargs
        )
        
        # Load concepts table
        self.concepts_table_path = self.dataset_path / 'concepts_table.csv'
        if self.concepts_table_path.exists():
            self.concepts_df = pd.read_csv(self.concepts_table_path)
            print(f"Loaded concepts_table.csv with {len(self.concepts_df)} entries")
            
            # Validate attacked_artifact exists in concepts_table
            if self.attacked_artifact is not None:
                available_artifacts = [col for col in self.concepts_df.columns 
                                      if col not in ['image', 'datasplit'] and not col.startswith('lbl-')]
                if self.attacked_artifact not in available_artifacts:
                    raise ValueError(f"Artifact '{self.attacked_artifact}' not found in concepts_table.csv. "
                                   f"Available artifacts: {available_artifacts}")
        else:
            self.concepts_df = None
            if self.attacked_artifact is not None:
                raise ValueError(f"concepts_table.csv not found at {self.concepts_table_path}. "
                               f"Required for artifact-to-artifact attack mode.")
            print(f"Warning: concepts_table.csv not found at {self.concepts_table_path}")
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'isic': {
                'image_extensions': ['.jpg', '.jpeg', '.png'],
                'class_columns': ['lbl-MEL', 'lbl-NV', 'lbl-BCC', 'lbl-AK', 'lbl-BKL', 
                                 'lbl-DF', 'lbl-VASC', 'lbl-SCC', 'lbl-UNK'],
            },
            'chexpert': {
                'image_extensions': ['.jpg', '.jpeg', '.png'],
            },
            'hyper_kvasir': {
                'image_extensions': ['.jpg', '.jpeg', '.png'],
            },
            'bone_age': {
                'image_extensions': ['.png', '.jpg', '.jpeg'],
            },
            'celeba': {
                'image_extensions': ['.jpg', '.jpeg', '.png'],
            },
            'imagenet': {
                'image_extensions': ['.jpg', '.jpeg', '.png', '.JPEG'],
            },
            'generic': {
                'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            },
        }
        
        if self.dataset_name in self.dataset_configs:
            self.config = self.dataset_configs[self.dataset_name]
        else:
            print(f"Warning: Unknown dataset '{dataset_name}'. Using generic configuration.")
            self.config = self.dataset_configs['generic']
    
    def load_dataset_metadata(self) -> Dict:
        """
        Load dataset metadata from concepts_table.csv.
        
        Returns:
            Dictionary with 'images' (list of paths), 'labels' (list of labels),
            and 'image_ids' (list of image IDs without extension)
        """
        images = []
        labels = []
        image_ids = []
        
        # If no concepts_table.csv was found, fall back to the ISIC ground truth CSV.
        # Rename raw class columns (MEL, NV, …) to lbl-MEL, lbl-NV, … so the rest
        # of this method works without modification.
        if self.concepts_df is None:
            gt_filenames = ['ISIC_2019_Training_GroundTruth.csv', 'ISIC_2020_Training_GroundTruth.csv']
            for gt_filename in gt_filenames:
                gt_path = self.dataset_path / gt_filename
                if gt_path.exists():
                    df_gt = pd.read_csv(gt_path)
                    label_cols = [c for c in df_gt.columns if c != 'image']
                    df_gt = df_gt.rename(columns={c: f'lbl-{c}' for c in label_cols})
                    self.concepts_df = df_gt
                    print(f"No concepts_table.csv found — using {gt_filename} ({len(df_gt)} rows)")
                    break
            if self.concepts_df is None:
                raise FileNotFoundError(
                    f"Neither concepts_table.csv nor any ground truth CSV found in {self.dataset_path}. "
                    f"Run 01_prepare_data.ipynb first, or ensure the ground truth CSV is present."
                )

        # Use concepts_table.csv (or the ground truth CSV loaded above) to find image paths and labels
        class_columns = self.config.get('class_columns',
            [col for col in self.concepts_df.columns if col.startswith('lbl-')])

        for _, row in self.concepts_df.iterrows():
            image_id = row['image']
            
            # Find the class label
            label = None
            for i, col in enumerate(class_columns):
                if col in row and row[col] == True:
                    label = i
                    break
            
            if label is None:
                label = 0  # Default label if not found
            
            # Find the image file
            for ext in self.config['image_extensions']:
                img_path = self.dataset_path / "Train" / f"{image_id}{ext}"
                if img_path.exists():
                    images.append(img_path)
                    labels.append(label)
                    image_ids.append(image_id)
                    break
    
        return {'images': images, 'labels': labels, 'image_ids': image_ids}
    
    def should_add_artifact(self, image_id: str, label: int) -> bool:
        """
        Determine if an artifact should be added to this sample.
        
        Args:
            image_id: The image identifier (without extension)
            label: The class label of the image
            
        Returns:
            True if artifact should be added, False otherwise
        """
        if self.attack_mode == 'class':
            # Class-based attack: check if label is in attacked_classes
            if label in self.attacked_classes:
                return self.rng.random() < self.p_artifact
            return False
        
        else:  # artifact-to-artifact mode
            # Check if the sample has the attacked_artifact
            if self.concepts_df is not None:
                row = self.concepts_df[self.concepts_df['image'] == image_id]
                if len(row) > 0 and self.attacked_artifact in row.columns:
                    has_artifact = row[self.attacked_artifact].values[0]
                    if has_artifact:
                        return self.rng.random() < self.p_artifact_to_artifact
            return False
    
    def generate(self, output_path) -> Path:
        """
        Generate the attacked dataset.
        
        Args:
            output_path: output directory
            
        Returns:
            Path to the generated attacked dataset
        """
        # Load metadata
        print(f"Loading dataset metadata from {self.dataset_path}...")
        metadata = self.load_dataset_metadata()
        images = metadata['images']
        labels = metadata['labels']
        image_ids = metadata['image_ids']
        
        print(f"Found {len(images)} images")
        
        if len(images) == 0:
            raise ValueError(f"No images found in {self.dataset_path}")

        output_path = Path(output_path)
        print(f"Creating attacked dataset at {output_path}")
        
        # Create output directory structure matching input
        images_output_path = output_path / "Train"
        images_output_path.mkdir(parents=True, exist_ok=True)

        # Copy ground truth CSV so ISICDataset can load this dir without needing parent fallback
        for gt_filename in ['ISIC_2019_Training_GroundTruth.csv', 'ISIC_2020_Training_GroundTruth.csv']:
            src_gt = self.dataset_path / gt_filename
            if src_gt.exists():
                shutil.copy2(src_gt, output_path / gt_filename)
                print(f"Copied {gt_filename} → {output_path}")

        # Copy split_ids.json to enforce the same val/test as the clean dataset
        src_split = self.dataset_path / 'split_ids.json'
        if src_split.exists():
            shutil.copy2(src_split, output_path / 'split_ids.json')
            print(f"Copied split_ids.json → {output_path}")
        else:
            print(f"Warning: split_ids.json not found at {src_split}. "
                  f"Run 01_prepare_data.ipynb first to generate it.")

        # Create a copy of concepts_df to update
        if self.concepts_df is not None:
            updated_concepts_df = self.concepts_df.copy()
            
            # Add column for new artifact if it doesn't exist
            if self.artifact_type not in updated_concepts_df.columns:
                updated_concepts_df[self.artifact_type] = False
        else:
            updated_concepts_df = None
        
        # Track statistics
        total_with_artifact = 0
        
        # Process images
        print(f"Generating dataset with artifact '{self.artifact_type}'...")
        if self.attack_mode == 'class':
            print(f"Altering correlation between: \t output   {str(self.attacked_classes):<20} \t&\t artifact {self.artifact_type}")
            print(f"Label-to-artifact probability: {self.p_artifact}")
        else:
            print(f"Altering correlation between: \t artifact {str(self.attacked_artifact):<20} \t&\t artifact {self.artifact_type}")
            print(f"Artifact-to-artifact probability: {self.p_artifact_to_artifact}")
        
        for i, (img_path, label, image_id) in tqdm(enumerate(zip(images, labels, image_ids)), 
                                                    mininterval=5.0, total=len(images), desc="Generating image samples:"):
            
            # Output path for image
            out_img_path = images_output_path / img_path.name
            
            # Check if this sample should have an artifact
            should_add = self.should_add_artifact(image_id, label)
            
            if should_add:
                # Load and process image
                try:
                    image = Image.open(img_path).convert('RGB')
                    modified_image, mask = self.artifact_generator.apply(image)
                    modified_image.save(out_img_path)
                    
                    # Update concepts_df
                    if updated_concepts_df is not None:
                        idx = updated_concepts_df[updated_concepts_df['image'] == image_id].index
                        if len(idx) > 0:
                            updated_concepts_df.loc[idx, self.artifact_type] = True
                    
                    total_with_artifact += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    shutil.copy2(img_path, out_img_path)
            else:
                # Copy original image
                shutil.copy2(img_path, out_img_path)
        
        # Save updated concepts_table.csv
        if updated_concepts_df is not None:
            concepts_output_path = output_path / 'concepts_table.csv'
            updated_concepts_df.to_csv(concepts_output_path, index=False)
            print(f"Saved updated concepts_table.csv to {concepts_output_path}")
        
        print(f"\nAttacked dataset generation complete!")
        print(f"Output directory: {output_path}")
        print(f"Images directory: {images_output_path}")
        print(f"Total images: {len(images)}")
        print(f"Images with artifact: {total_with_artifact}")
        
        return output_path


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate attacked dataset with artificial artifacts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Class-based attack: Add timestamp to 90% of class 0 samples
python generate_attacked_dataset.py \\
    --dataset_name isic \\
    --dataset_path ./datasets/ \\
    --attacked_classes 0 \\
    --artifact_type timestamp \\
    --p_artifact 0.9

# Artifact-to-artifact attack: Add microscope to 40% of samples with red_color
python generate_attacked_dataset.py \\
    --dataset_name isic \\
    --dataset_path ./datasets/ \\
    --attacked_artifact red_color \\
    --artifact_type microscope \\
    --p_artifact_to_artifact 0.4

# Multiple classes attack:
python generate_attacked_dataset.py \\
    --dataset_name isic \\
    --dataset_path ./datasets/ \\
    --attacked_classes 0 1 2 \\
    --artifact_type brightness \\
    --p_artifact 0.8
        """
    )
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Name of the dataset (isic, chexpert, hyper_kvasir, bone_age, celeba, imagenet, generic)'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to the original dataset images'
    )
    
    parser.add_argument(
        '--artifact_type',
        type=str,
        required=True,
        choices=['timestamp', 'microscope', 'brightness', 'lsb', 'noise', 
                 'colored_square', 'ruler', 'band_aid', 'circle', 'text', 'blur'],
        help='Type of artifact to add'
    )
    
    # Class-based attack arguments
    parser.add_argument(
        '--attacked_classes',
        type=int,
        nargs='+',
        default=None,
        help='Class indices to add artifacts to (for class-based attack)'
    )
    
    parser.add_argument(
        '--p_artifact',
        type=float,
        default=0.9,
        help='Proportion of attacked class samples to add artifact to (default: 0.9)'
    )
    
    # Artifact-to-artifact attack arguments
    parser.add_argument(
        '--attacked_artifact',
        type=str,
        default=None,
        help='Name of existing artifact to target (for artifact-to-artifact attack). '
             'Must exist in concepts_table.csv'
    )
    
    parser.add_argument(
        '--p_artifact_to_artifact',
        type=float,
        default=None,
        help='Proportion of samples with existing artifact to add new artifact to '
             '(required for artifact-to-artifact attack)'
    )
    
    # Common arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: same as dataset parent directory)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--artifact_kwargs',
        type=str,
        default='{}',
        help='JSON string of additional artifact parameters'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if args.attacked_classes is not None and args.attacked_artifact is not None:
        raise ValueError("Cannot specify both --attacked_classes and --attacked_artifact. Choose one attack mode.")
    
    if args.attacked_classes is None and args.attacked_artifact is None:
        raise ValueError("Must specify either --attacked_classes or --attacked_artifact.")
    
    if args.attacked_artifact is not None and args.p_artifact_to_artifact is None:
        raise ValueError("When using --attacked_artifact, must also specify --p_artifact_to_artifact.")
    
    # Parse artifact kwargs
    try:
        artifact_kwargs = json.loads(args.artifact_kwargs)
    except json.JSONDecodeError as e:
        print(f"Error parsing artifact_kwargs: {e}")
        artifact_kwargs = {}
    
    # Create generator
    generator = AttackedDatasetGenerator(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        artifact_type=args.artifact_type,
        attacked_classes=args.attacked_classes,
        p_artifact=args.p_artifact,
        attacked_artifact=args.attacked_artifact,
        p_artifact_to_artifact=args.p_artifact_to_artifact,
        seed=args.seed,
        **artifact_kwargs
    )
    if args.output_dir is None: 
        print(f"[WARN] No output_dir specified, saving attacked dataset to same location as input dataset with a custom folder name: {args.dataset_path}")
        output_dir_name = f"{args.artifact_type}-on-{args.attacked_artifact}-{int(args.p_artifact_to_artifact*100):02d}" if args.attacked_artifact else f"{args.artifact_type}-{int(args.p_artifact*100):02d}"
        output_path =  os.path.join(args.dataset_path, output_dir_name)

    # Generate attacked dataset
    output_path = generator.generate(output_path=args.output_dir)
    
    print(f"\nDone! Attacked dataset saved to: {output_path}")


if __name__ == '__main__':
    main()