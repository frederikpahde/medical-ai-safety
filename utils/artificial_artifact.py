import math
from datetime import datetime
from datetime import timedelta
import cv2

import numpy as np
import torch
import torchvision.transforms as T
from PIL import ImageDraw, Image


def get_artifact_kwargs(config):
    artifact_kwargs = {}
    artifact_type = config.get("artifact_type", None)
    if artifact_type == "channel":
        artifact_kwargs = {
            'op_type': config.get('op_type', 'add'),
            'channel': config.get('channel', 0),
            'value': config.get('value', 100)
        }
    elif artifact_type == "lsb":
        artifact_kwargs = {
            'lsb_trigger': config.get('lsb_trigger', "ThisIsASecretCleverHansTrigger"),
            'lsb_factor': config.get("lsb_factor", 3),
            'start_bit': config.get("start_bit", 0)
        }
    elif artifact_type == "random_mnist":
        artifact_kwargs = {
            'shift_factor': config['shift_factor'],
            'datapath_mnist': config['datapath_mnist']
        }
    elif artifact_type == "ch_time":
        artifact_kwargs = {
            "time_format": config.get("time_format", "time")
        }
    elif artifact_type == "color_mnist":
        artifact_kwargs = {
            "color_id": config["attacked_classes"][0]
        }
    elif artifact_type == "white_color":
        artifact_kwargs = {
            "alpha": config.get("alpha", .3)
        }
    elif artifact_type == "defective_lead":
        artifact_kwargs = {
            "lead_ids": config.get("lead_ids", [1]),
            "seq_length": config.get("seq_length", 100)
        }
    elif artifact_type == "amplified_beat":
        artifact_kwargs = {
            "lead_ids": config.get("lead_ids", [3]),
            "peak_ids": config.get("peak_ids", [2]),
            "amplify_factor": config.get("amplify_factor", 3)
        }
    return artifact_kwargs


def insert_artifact(img, artifact_type, **kwargs):
    if artifact_type == "ch_time":
        return insert_artifact_ch_time(img, **kwargs)
    elif artifact_type == "ch_text":
        return insert_artifact_ch_text(img, **kwargs)
    elif artifact_type == "microscope":
        return insert_microscope(img, **kwargs)
    elif artifact_type == "channel":
        return insert_artifact_channel(img, **kwargs)
    elif artifact_type == "white_color":
        return insert_artifact_white_color(img, **kwargs)
    elif artifact_type == "red_color":
        return insert_artifact_red_color(img, **kwargs)
    elif artifact_type == "lsb":
        return insert_lsb_trigger(img, **kwargs)
    elif artifact_type == "constant_box":
        return insert_constant_box(img, **kwargs)
    elif artifact_type == "random_box":
        return insert_random_box(img, **kwargs)
    elif artifact_type == "random_mnist":
        return insert_random_mnist(img, **kwargs)
    elif artifact_type == "color_mnist":
        return color_digit(img, **kwargs)
    else:
        raise ValueError(f"Unknown artifact_type: {artifact_type}")

def color_digit(img, **kwargs):
    assert "color_id" in kwargs
    color_id = kwargs["color_id"]
    COLOR_MAP = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 128, 255),
        (0, 0, 255),
        (127, 0, 255),
        (255, 0, 255),
        (255, 0, 127)
    ]

    color = COLOR_MAP[color_id]
    img_np = np.array(img)
    img_corrupted = (img_np * color / 255).round().astype(np.uint8)
    mask = img_np[0].round().squeeze()
    return Image.fromarray(img_corrupted), mask

def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = np.random.randint(int_delta)
    return start + timedelta(seconds=random_second)

def insert_artifact_ch_time(img, **kwargs):
    time_format = kwargs.get("time_format", "datetime")
    time_only = time_format == "time"
    d1 = datetime.strptime('01/01/2020', '%m/%d/%Y')
    d2 = datetime.strptime('12/31/2022', '%m/%d/%Y')
    kwargs["reserved_length"] = 60 if time_only else 100
    date = random_date(d1, d2)
    if time_only:
        kwargs["min_val"] = 125
        kwargs["max_val"] = 0
        date = date.strftime("%H:%M:%S")
    kwargs["text"] = str(date)
    color = (
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255), 
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255), 
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255)
    )
    kwargs["color"] = color

    return insert_artifact_ch_text(img, **kwargs)

def insert_random_mnist(img, **kwargs):
    shift_factor = kwargs['shift_factor']
    data_mnist = kwargs['data_mnist']
    random_idx = np.random.choice(len(data_mnist))
    
    img = np.array(img)
    transforms = T.Compose([T.Resize(img.shape[1]), T.ToTensor()])
    img_mnist = transforms(data_mnist[random_idx][0])

    img = img + shift_factor * (np.moveaxis(img_mnist.numpy(), 0, 2) * 255.)
    img = np.clip(img, 0, 255.).astype(np.uint8)
    mask = img_mnist.round().squeeze()
    return Image.fromarray(img), mask


def insert_constant_box(img, **kwargs):
    img = np.array(img)
    size = kwargs.get('size', 2)
    offset = kwargs.get('offset', 1)
    img[offset:offset + size, offset:offset + size, :] = 255
    mask = torch.zeros(img.shape[:2])
    mask[offset:offset + size, offset:offset + size] = 1

    return Image.fromarray(img), mask


def insert_random_box(img, **kwargs):
    size = np.random.randint(1, 5)
    img = np.array(img)
    posx, posy = np.random.randint(1, img.shape[0] - (size + 1)), np.random.randint(1, img.shape[1] - (size + 1))
    img[posx:posx + size, posy:posy + size, :] = 255 - np.random.rand() * .1 * 255

    mask = torch.zeros(img.shape[:2])
    mask[posx:posx + size, posy:posy + size] = 1
    return Image.fromarray(img), mask


def insert_lsb_trigger(img, **kwargs):
    text_trigger = kwargs.get('lsb_trigger',
                              "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat.   Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi.   Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis.   At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no rebum. sanctus sea sed takimata ut vero voluptua. est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur")
    lsb_factor = kwargs.get("lsb_factor", 3)
    start_bit = kwargs.get("start_bit", 0)
    img = np.array(img)
    shape = img.shape
    img_new = img.copy().reshape(-1)

    assert lsb_factor < 8, f"LSB factor has to be <8 (is: {lsb_factor})"

    b_message = ''.join([format(ord(i), "08b")[start_bit:] for i in text_trigger])
    multiplicator = math.ceil(len(img_new) / len(b_message) * lsb_factor)
    # print("multiplicator", multiplicator)
    b_message *= multiplicator
    b_message = b_message[:(len(img_new) * lsb_factor)]
    b_message_int = [int(c) for c in b_message]

    img_new_b = np.unpackbits(img_new.reshape(len(img_new), 1), axis=1)
    for bit_index in range(1, lsb_factor+1):
        ind_start, ind_end = (bit_index-1)*len(img_new), bit_index*len(img_new)
        b_message_chunk = b_message_int[ind_start:ind_end]
        img_new_b[:, -bit_index] = b_message_chunk

    img_new = np.packbits(img_new_b, axis=1).reshape(-1)
    img_new = Image.fromarray(img_new.reshape(shape))
    
    mask = torch.ones((img.shape[0], img.shape[1]))
    return img_new, mask


def insert_artifact_white_color(img, **kwargs):
    img = np.array(img).astype(np.float64)
    alpha = kwargs.get("alpha", 0.3)
    img[:, :, 0] = img[:, :, 0] * (1 - alpha) + alpha * 255
    img[:, :, 1] = img[:, :, 1] * (1 - alpha) + alpha * 255
    img[:, :, 2] = img[:, :, 2] * (1 - alpha) + alpha * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)

    return img, mask


def insert_artifact_red_color(img, **kwargs):
    img = np.array(img).astype(np.float64)
    alpha = 0.2
    img[:, :, 0] = img[:, :, 0] * (1 - alpha) + alpha * 255
    img[:, :, 1] = img[:, :, 1] * (1 - alpha) + alpha * 0
    img[:, :, 2] = img[:, :, 2] * (1 - alpha) + alpha * 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)
    return img, mask


def insert_artifact_channel(img, **kwargs):
    img = np.array(img).astype(np.float64)

    op_type = kwargs.get("op_type", "add")
    channel = kwargs.get("channel", 0)
    value = kwargs.get("value", 100)

    if op_type == "const":
        img[:, :, channel] = value
    elif op_type == "add":
        img[:, :, channel] += value
    elif op_type == "mul":
        img[:, :, channel] *= value
    else:
        raise ValueError(f"Unknown op_type '{op_type}', choose one of 'mul', 'add', 'const'")

    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)

    return img, mask

def insert_microscope(img, **kwargs):
    img = np.array(img).astype(np.float64)
    circle = cv2.circle(
                (np.ones(img.shape)).astype(np.uint8),
                (img.shape[0]//2, img.shape[1]//2),
                np.random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                (0, 0, 0),
                -1
        )
    mask = circle
    img = np.multiply(img, 1-mask)
    img = Image.fromarray(img.astype(np.uint8))
    return img, torch.Tensor(mask[:,:,0])

def insert_artifact_ch_text(img, **kwargs):
    text = kwargs.get("text", "Clever Hans")
    fill = kwargs.get("fill", (0, 0, 0))
    img_size = kwargs.get("img_size", 224)
    color = kwargs.get("color", (255, 255, 255))
    reserved_length = kwargs.get("reserved_length", 80)
    min_val = kwargs.get("min_val", 25)
    max_val = kwargs.get("max_val", 25)
    padding = 15

    # Random position
    end_x = img_size - reserved_length
    end_y = img_size - 20
    valid_positions = np.array([
        [padding + 5, padding + 5], 
        [padding + 5, end_y - padding - 5], 
        [end_x - padding - 5, padding + 5], 
        [end_x - padding - 5, end_y - padding - 5]
    ])
    pos = valid_positions[np.random.choice(len(valid_positions))]
    pos += np.random.normal(0, 2, 2).astype(int)
    pos[0] = np.clip(pos[0], padding, end_x - padding)
    pos[1] = np.clip(pos[1], padding, end_y - padding)

    # Random size
    size_text_img = np.random.choice(np.arange(img_size - min_val, img_size + max_val))

    # Scale pos
    scaling = size_text_img / img_size
    pos = tuple((int(pos[0] * scaling), int(pos[1] * scaling)))

    # Add Random Noise to color
    fill = tuple(np.clip(np.array(fill) + np.random.normal(0, 10, 3), 0, 255).astype(int))
    
    # Random Rotation
    rotation = np.random.choice(np.arange(-30, 31) / 10)
    image_text = Image.new('RGBA', (size_text_img, size_text_img), (0,0,0,0))
    draw = ImageDraw.Draw(image_text)
    draw.text(pos, text=text, fill=color)
    image_text = T.Resize((img_size, img_size))(image_text.rotate(rotation))

    # Insert text into image
    out = Image.composite(image_text, img, image_text)

    mask = torch.zeros((img_size, img_size))
    mask_coord = image_text.getbbox()
    mask[mask_coord[1]:mask_coord[3], mask_coord[0]:mask_coord[2]] = 1

    return out, mask



"""
Artificial Artifact Utilities
==============================
This module provides functions for adding artificial artifacts to images,
designed to be compatible with the medical-ai-safety repository structure.

This file should be placed at: utils/artificial_artifact.py

The module supports various artifact types used in medical AI safety research
to study and mitigate spurious correlations in deep learning models.

Reference:
- Pahde et al., "Ensuring Medical AI Safety: Explainable AI-Driven Detection 
  and Mitigation of Spurious Model Behavior and Associated Data"
- https://github.com/frederikpahde/medical-ai-safety
"""

import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Type aliases
ImageType = Union[Image.Image, np.ndarray]
MaskType = np.ndarray


def to_pil(image: ImageType) -> Image.Image:
    """Convert numpy array to PIL Image if needed."""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return image.copy()


def to_numpy(image: ImageType) -> np.ndarray:
    """Convert PIL Image to numpy array if needed."""
    if isinstance(image, Image.Image):
        return np.array(image)
    return image.copy()


# =============================================================================
# Timestamp Artifact (commonly used in HyperKvasir)
# =============================================================================

def add_timestamp(
    image: ImageType,
    timestamp_text: Optional[str] = None,
    position: str = 'top_left',
    font_size: Optional[int] = None,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 0),
    padding: int = 10,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[Image.Image, MaskType]:
    """
    Add a timestamp text overlay to the image.
    
    This simulates timestamps commonly found in medical imaging devices
    like endoscopy equipment (e.g., HyperKvasir dataset).
    
    Args:
        image: Input image (PIL Image or numpy array)
        timestamp_text: Custom timestamp text (auto-generated if None)
        position: Position of timestamp ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        font_size: Font size (auto-calculated if None)
        bg_color: Background color (RGB tuple)
        text_color: Text color (RGB tuple)
        padding: Padding from image edge
        rng: Random state for reproducibility
        
    Returns:
        Tuple of (modified_image, artifact_mask)
    """
    image = to_pil(image)
    width, height = image.size
    
    if rng is None:
        rng = np.random.RandomState()
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Generate timestamp text if not provided
    if timestamp_text is None:
        year = rng.randint(2015, 2024)
        month = rng.randint(1, 13)
        day = rng.randint(1, 29)
        hour = rng.randint(0, 24)
        minute = rng.randint(0, 60)
        timestamp_text = f"{year}/{month:02d}/{day:02d} {hour:02d}:{minute:02d}"
    
    # Create drawing context
    draw = ImageDraw.Draw(image)
    
    # Font size
    if font_size is None:
        font_size = max(12, min(width, height) // 20)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), timestamp_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    positions = {
        'top_left': (padding, padding),
        'top_right': (width - text_width - padding, padding),
        'bottom_left': (padding, height - text_height - padding),
        'bottom_right': (width - text_width - padding, height - text_height - padding),
    }
    x, y = positions.get(position, positions['top_left'])
    
    # Draw background rectangle and text
    draw.rectangle([x - 2, y - 2, x + text_width + 2, y + text_height + 2], fill=bg_color)
    draw.text((x, y), timestamp_text, font=font, fill=text_color)
    
    # Update mask
    mask[max(0, y-2):min(height, y+text_height+2), max(0, x-2):min(width, x+text_width+2)] = 255
    
    return image, mask


# =============================================================================
# Microscope Border Artifact (dermoscopy simulation)
# =============================================================================

def add_microscope_border(
    image: ImageType,
    radius: Optional[int] = None,
    border_color: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[Image.Image, MaskType]:
    """
    Add a black circular border simulating a microscope/dermoscope view.
    
    This is commonly seen in dermoscopic images where a circular
    field of view is visible.
    
    Args:
        image: Input image (PIL Image or numpy array)
        radius: Radius of visible area (auto-calculated if None)
        border_color: Color of the border region (RGB tuple)
        
    Returns:
        Tuple of (modified_image, artifact_mask)
    """
    image = to_pil(image)
    width, height = image.size
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate center and radius
    center_x, center_y = width // 2, height // 2
    if radius is None:
        radius = min(width, height) // 2 - 10
    
    # Create masked array
    img_array = np.array(image)
    
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist > radius:
                img_array[y, x] = border_color if len(img_array.shape) == 3 else border_color[0]
                mask[y, x] = 255
    
    return Image.fromarray(img_array), mask


# =============================================================================
# Brightness Artifact (CheXpert-style)
# =============================================================================

def increase_brightness(
    image: ImageType,
    brightness_factor: float = 1.5
) -> Tuple[Image.Image, MaskType]:
    """
    Increase image brightness uniformly.
    
    This simulates over-exposure or brightness artifacts common in
    chest X-rays and other radiological images (e.g., CheXpert).
    
    Args:
        image: Input image (PIL Image or numpy array)
        brightness_factor: Multiplier for brightness (>1 increases brightness)
        
    Returns:
        Tuple of (modified_image, full_mask)
    """
    image = to_pil(image)
    width, height = image.size
    
    # Create full mask (entire image affected)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Apply brightness enhancement
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    
    return image, mask


# =============================================================================
# LSB Watermark Artifact (ISIC-style)
# =============================================================================

def add_lsb_watermark(
    image: ImageType,
    pattern_size: int = 8,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[Image.Image, MaskType]:
    """
    Add LSB (Least Significant Bit) watermark to the image.
    
    This creates an invisible watermark that can be learned by neural
    networks, creating a shortcut for classification.
    
    Args:
        image: Input image (PIL Image or numpy array)
        pattern_size: Size of the repeating pattern
        rng: Random state for reproducibility
        
    Returns:
        Tuple of (modified_image, full_mask)
    """
    img_array = to_numpy(image)
    height, width = img_array.shape[:2]
    
    if rng is None:
        rng = np.random.RandomState()
    
    # Create full mask
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Create repeating pattern
    pattern = rng.randint(0, 2, (pattern_size, pattern_size)).astype(np.uint8)
    
    # Tile pattern to image size
    full_pattern = np.tile(pattern, (height // pattern_size + 1, width // pattern_size + 1))
    full_pattern = full_pattern[:height, :width]
    
    # Apply LSB modification
    if len(img_array.shape) == 3:
        for c in range(img_array.shape[2]):
            img_array[:, :, c] = (img_array[:, :, c] & 0xFE) | full_pattern
    else:
        img_array = (img_array & 0xFE) | full_pattern
    
    return Image.fromarray(img_array), mask


# =============================================================================
# Static Noise Artifact (PTB-XL-style)
# =============================================================================

def add_static_noise(
    image: ImageType,
    noise_level: float = 25,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[Image.Image, MaskType]:
    """
    Add static noise pattern to the image.
    
    For ECG data (1D signals), this adds a consistent noise pattern
    to a specific lead. For images, it adds visual noise.
    
    Args:
        image: Input image (PIL Image or numpy array)
        noise_level: Standard deviation of Gaussian noise
        rng: Random state for reproducibility
        
    Returns:
        Tuple of (modified_image, full_mask)
    """
    img_array = to_numpy(image).astype(np.float32)
    height, width = img_array.shape[:2]
    
    if rng is None:
        rng = np.random.RandomState()
    
    # Create full mask
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Generate and add noise
    noise = rng.randn(*img_array.shape) * noise_level
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array), mask


# =============================================================================
# Colored Square Artifact
# =============================================================================

def add_colored_square(
    image: ImageType,
    square_size: Optional[int] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    corner: str = 'bottom_right'
) -> Tuple[Image.Image, MaskType]:
    """
    Add a colored square to a corner of the image.
    
    This is a simple but effective artifact that creates strong
    spurious correlations in classification models.
    
    Args:
        image: Input image (PIL Image or numpy array)
        square_size: Size of the square (auto-calculated if None)
        color: Color of the square (RGB tuple)
        corner: Corner position ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        
    Returns:
        Tuple of (modified_image, artifact_mask)
    """
    image = to_pil(image)
    width, height = image.size
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate square size
    if square_size is None:
        square_size = min(width, height) // 10
    
    # Calculate position
    positions = {
        'top_left': (0, 0),
        'top_right': (width - square_size, 0),
        'bottom_left': (0, height - square_size),
        'bottom_right': (width - square_size, height - square_size),
    }
    x1, y1 = positions.get(corner, positions['bottom_right'])
    x2, y2 = x1 + square_size, y1 + square_size
    
    # Draw square
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Update mask
    mask[y1:y2, x1:x2] = 255
    
    return image, mask


# =============================================================================
# Ruler Artifact (ISIC-style)
# =============================================================================

def add_ruler(
    image: ImageType,
    ruler_width: Optional[int] = None,
    ruler_color: Tuple[int, int, int] = (0, 0, 0),
    tick_color: Tuple[int, int, int] = (255, 255, 255),
    position: str = 'bottom'
) -> Tuple[Image.Image, MaskType]:
    """
    Add a ruler-like artifact to the image edge.
    
    This simulates the ruler artifacts commonly found in dermoscopic
    images of the ISIC dataset.
    
    Args:
        image: Input image (PIL Image or numpy array)
        ruler_width: Width of the ruler (auto-calculated if None)
        ruler_color: Background color of ruler (RGB tuple)
        tick_color: Color of tick marks (RGB tuple)
        position: Position ('bottom', 'right', 'top', 'left')
        
    Returns:
        Tuple of (modified_image, artifact_mask)
    """
    image = to_pil(image)
    width, height = image.size
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    draw = ImageDraw.Draw(image)
    
    # Calculate ruler width
    if ruler_width is None:
        ruler_width = max(10, min(width, height) // 20)
    
    if position == 'bottom':
        draw.rectangle([0, height - ruler_width, width, height], fill=ruler_color)
        mask[height - ruler_width:height, :] = 255
        
        tick_spacing = max(1, width // 20)
        for i in range(0, width, tick_spacing):
            tick_height = ruler_width // 2 if i % (tick_spacing * 2) == 0 else ruler_width // 4
            draw.line([(i, height - tick_height), (i, height)], fill=tick_color, width=2)
    
    elif position == 'right':
        draw.rectangle([width - ruler_width, 0, width, height], fill=ruler_color)
        mask[:, width - ruler_width:width] = 255
        
        tick_spacing = max(1, height // 20)
        for i in range(0, height, tick_spacing):
            tick_width = ruler_width // 2 if i % (tick_spacing * 2) == 0 else ruler_width // 4
            draw.line([(width - tick_width, i), (width, i)], fill=tick_color, width=2)
    
    elif position == 'top':
        draw.rectangle([0, 0, width, ruler_width], fill=ruler_color)
        mask[0:ruler_width, :] = 255
        
        tick_spacing = max(1, width // 20)
        for i in range(0, width, tick_spacing):
            tick_height = ruler_width // 2 if i % (tick_spacing * 2) == 0 else ruler_width // 4
            draw.line([(i, 0), (i, tick_height)], fill=tick_color, width=2)
    
    elif position == 'left':
        draw.rectangle([0, 0, ruler_width, height], fill=ruler_color)
        mask[:, 0:ruler_width] = 255
        
        tick_spacing = max(1, height // 20)
        for i in range(0, height, tick_spacing):
            tick_width = ruler_width // 2 if i % (tick_spacing * 2) == 0 else ruler_width // 4
            draw.line([(0, i), (tick_width, i)], fill=tick_color, width=2)
    
    return image, mask


# =============================================================================
# Band-Aid Artifact (ISIC-style)
# =============================================================================

def add_band_aid(
    image: ImageType,
    band_length: Optional[int] = None,
    band_width: Optional[int] = None,
    band_color: Tuple[int, int, int] = (210, 180, 140),
    center_x: Optional[int] = None,
    center_y: Optional[int] = None,
    angle: Optional[float] = None,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[Image.Image, MaskType]:
    """
    Add a band-aid-like artifact to the image.
    
    This simulates the band-aid artifacts found in dermoscopic images
    of the ISIC dataset.
    
    Args:
        image: Input image (PIL Image or numpy array)
        band_length: Length of band-aid (auto-calculated if None)
        band_width: Width of band-aid (auto-calculated if None)
        band_color: Color of band-aid (RGB tuple)
        center_x: X coordinate of center (random if None)
        center_y: Y coordinate of center (random if None)
        angle: Rotation angle in degrees (random if None)
        rng: Random state for reproducibility
        
    Returns:
        Tuple of (modified_image, artifact_mask)
    """
    image = to_pil(image)
    width, height = image.size
    
    if rng is None:
        rng = np.random.RandomState()
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate dimensions
    if band_length is None:
        band_length = min(width, height) // 3
    if band_width is None:
        band_width = band_length // 6
    
    # Calculate position and angle
    if center_x is None:
        center_x = rng.randint(band_length // 2, width - band_length // 2)
    if center_y is None:
        center_y = rng.randint(band_width, height - band_width)
    if angle is None:
        angle = rng.uniform(-30, 30)
    
    # Create band-aid on separate image
    band_aid = Image.new('RGBA', (band_length, band_width), (0, 0, 0, 0))
    band_draw = ImageDraw.Draw(band_aid)
    band_draw.rounded_rectangle([0, 0, band_length-1, band_width-1], 
                                radius=band_width//2, fill=band_color + (255,))
    
    # Add perforations
    perf_color = (band_color[0] - 20, band_color[1] - 20, band_color[2] - 20, 255)
    for i in range(5, band_length - 5, 8):
        for j in range(2, band_width - 2, 4):
            band_draw.ellipse([i-1, j-1, i+1, j+1], fill=perf_color)
    
    # Rotate
    band_aid = band_aid.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Calculate paste position
    paste_x = center_x - band_aid.width // 2
    paste_y = center_y - band_aid.height // 2
    
    # Paste with transparency
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    image.paste(band_aid, (paste_x, paste_y), band_aid)
    image = image.convert('RGB')
    
    # Update mask
    mask_band = np.array(band_aid.split()[-1])
    mask_band = (mask_band > 0).astype(np.uint8) * 255
    
    py1, py2 = max(0, paste_y), min(height, paste_y + band_aid.height)
    px1, px2 = max(0, paste_x), min(width, paste_x + band_aid.width)
    my1, my2 = max(0, -paste_y), min(band_aid.height, height - paste_y)
    mx1, mx2 = max(0, -paste_x), min(band_aid.width, width - paste_x)
    
    if py1 < py2 and px1 < px2:
        mask[py1:py2, px1:px2] = mask_band[my1:my2, mx1:mx2]
    
    return image, mask


# =============================================================================
# Skin Marker Artifact (ISIC-style)
# =============================================================================

def add_skin_marker(
    image: ImageType,
    marker_type: str = 'dot',
    marker_color: Tuple[int, int, int] = (0, 0, 255),
    size: Optional[int] = None,
    position: Optional[Tuple[int, int]] = None,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[Image.Image, MaskType]:
    """
    Add a skin marker artifact to the image.
    
    This simulates skin markers commonly found in dermoscopic images
    of the ISIC dataset.
    
    Args:
        image: Input image (PIL Image or numpy array)
        marker_type: Type of marker ('dot', 'cross', 'circle')
        marker_color: Color of marker (RGB tuple)
        size: Size of marker (auto-calculated if None)
        position: Position as (x, y) tuple (random if None)
        rng: Random state for reproducibility
        
    Returns:
        Tuple of (modified_image, artifact_mask)
    """
    image = to_pil(image)
    width, height = image.size
    
    if rng is None:
        rng = np.random.RandomState()
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate size and position
    if size is None:
        size = min(width, height) // 20
    if position is None:
        position = (rng.randint(size, width - size), rng.randint(size, height - size))
    
    draw = ImageDraw.Draw(image)
    x, y = position
    
    if marker_type == 'dot':
        draw.ellipse([x - size, y - size, x + size, y + size], fill=marker_color)
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                if dx*dx + dy*dy <= size*size:
                    if 0 <= y + dy < height and 0 <= x + dx < width:
                        mask[y + dy, x + dx] = 255
    
    elif marker_type == 'cross':
        line_width = max(2, size // 3)
        draw.line([(x - size, y), (x + size, y)], fill=marker_color, width=line_width)
        draw.line([(x, y - size), (x, y + size)], fill=marker_color, width=line_width)
        mask[max(0, y - line_width//2):min(height, y + line_width//2 + 1), 
             max(0, x - size):min(width, x + size + 1)] = 255
        mask[max(0, y - size):min(height, y + size + 1), 
             max(0, x - line_width//2):min(width, x + line_width//2 + 1)] = 255
    
    elif marker_type == 'circle':
        line_width = max(2, size // 4)
        draw.ellipse([x - size, y - size, x + size, y + size], 
                     outline=marker_color, width=line_width)
        for dx in range(-size - line_width, size + line_width + 1):
            for dy in range(-size - line_width, size + line_width + 1):
                dist = np.sqrt(dx*dx + dy*dy)
                if abs(dist - size) <= line_width:
                    if 0 <= y + dy < height and 0 <= x + dx < width:
                        mask[y + dy, x + dx] = 255
    
    return image, mask


# =============================================================================
# Convenience function to get artifact by name
# =============================================================================

def get_artifact_function(artifact_type: str):
    """
    Get the artifact function by name.
    
    Args:
        artifact_type: Name of the artifact type
        
    Returns:
        Function that applies the artifact
    """
    functions = {
        'timestamp': add_timestamp,
        'microscope': add_microscope_border,
        'brightness': increase_brightness,
        'lsb': add_lsb_watermark,
        'noise': add_static_noise,
        'colored_square': add_colored_square,
        'ruler': add_ruler,
        'band_aid': add_band_aid,
        'skin_marker': add_skin_marker,
    }
    
    if artifact_type not in functions:
        raise ValueError(f"Unknown artifact type: {artifact_type}. "
                        f"Available: {list(functions.keys())}")
    
    return functions[artifact_type]


def apply_artifact(
    image: ImageType,
    artifact_type: str,
    **kwargs
) -> Tuple[Image.Image, MaskType]:
    """
    Apply an artifact to an image.
    
    Args:
        image: Input image
        artifact_type: Type of artifact to apply
        **kwargs: Additional arguments for the artifact function
        
    Returns:
        Tuple of (modified_image, artifact_mask)
    """
    func = get_artifact_function(artifact_type)
    return func(image, **kwargs)


# =============================================================================
# ECG-specific artifacts (for PTB-XL)
# =============================================================================

def add_ecg_static_noise(
    signal: np.ndarray,
    lead_idx: int = 0,
    noise_level: float = 0.1,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add static noise to a specific lead in ECG data.
    
    This is used for PTB-XL dataset experiments.
    
    Args:
        signal: ECG signal array of shape (n_leads, n_samples) or (n_samples,)
        lead_idx: Index of the lead to add noise to
        noise_level: Standard deviation of noise relative to signal
        rng: Random state for reproducibility
        
    Returns:
        Tuple of (modified_signal, noise_mask)
    """
    if rng is None:
        rng = np.random.RandomState()
    
    signal = signal.copy()
    
    if signal.ndim == 1:
        # Single lead
        noise = rng.randn(len(signal)) * noise_level * np.std(signal)
        signal = signal + noise
        mask = np.ones(len(signal), dtype=np.uint8) * 255
    else:
        # Multiple leads
        noise = rng.randn(signal.shape[1]) * noise_level * np.std(signal[lead_idx])
        signal[lead_idx] = signal[lead_idx] + noise
        mask = np.zeros(signal.shape, dtype=np.uint8)
        mask[lead_idx] = 255
    
    return signal, mask
