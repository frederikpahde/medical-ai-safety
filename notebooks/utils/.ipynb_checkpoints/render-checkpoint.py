import numpy as np
import torch
from PIL import ImageFilter, Image, ImageDraw
from crp.image import get_crop_range, imgify
from torchvision.transforms.functional import gaussian_blur
from zennit.core import stabilize
import torchvision

COLORS = [
    np.array((54, 197, 240)) / 255.,
    np.array((210, 40, 95)) / 255.,
    np.array((236, 178, 46)) / 255.,
    np.array((15, 157, 88)) / 255.,
    np.array((84, 25, 85)) / 255.
]


def overlay_heatmap(img, heatmap, color):
    gaussian = torchvision.transforms.GaussianBlur(kernel_size=71, sigma=21.0)
    heatmap = gaussian(heatmap[None, :, :])[0]
    heatmap_norm = heatmap / heatmap.abs().max()
    heatmap_np = heatmap_norm.numpy()
    heatmap_np[heatmap_np < np.percentile(heatmap_np, 30)] = 0
    # Convert img to numpy array for plotting
    img_np = img.permute(1, 2, 0).numpy()
    overlay = img_np.copy()

    # Create a color heatmap
    color_heatmap = np.zeros((3, 224, 224))
    for i in range(3):  # Apply heatmap to the specified color channels
        color_heatmap[i] = heatmap_np * color[i]

    heatmap_np_scaled = heatmap_np * .75
    # Overlay the color heatmap on the original image
    for i in range(3):  # For each color channel
        overlay[:, :, i] = img_np[:, :, i] * (1 - heatmap_np_scaled) + color_heatmap[i] * heatmap_np_scaled

    return overlay.clip(0,1)

def overlay_heatmaps(img, heatmaps, colors):
    assert len(heatmaps) == len(colors), f"same number of heatmaps and colors required, is {len(heatmaps)} / len({colors})"
    ## TODO: solve issue that earlier heatmaps fade out
    for hm, c in zip(heatmaps[::-1], colors[::-1]):
         img = torch.tensor(overlay_heatmap(img.detach().cpu(), hm, c).transpose(2,0,1))
    return img

@torch.no_grad()
def vis_opaque_img_border(data_batch, heatmaps, rf=False, alpha=0.5, vis_th=0.05, crop_th=0.01,
                          kernel_size=13) -> Image.Image:
    """
    Draws reference images. The function lowers the opacity in regions with relevance lower than max(relevance)*vis_th.
    In addition, the reference image can be cropped where relevance is less than max(relevance)*crop_th by setting 'rf' to True.

    Parameters:
    ----------
    data_batch: torch.Tensor
        original images from dataset without FeatureVisualization.preprocess() applied to it
    heatmaps: torch.Tensor
        ouput heatmap tensor of the CondAttribution call
    rf: boolean
        Computes the CRP heatmap for a single neuron and hence restricts the heatmap to the receptive field.
        The amount of cropping is further specified by the 'crop_th' argument.
    alpha: between [0 and 1]
        Regulates the transparency in low relevance regions.
    vis_th: between [0 and 1)
        Visualization Threshold: Increases transparency in regions where relevance is smaller than max(relevance)*vis_th.
    crop_th: between [0 and 1)
        Cropping Threshold: Crops the image in regions where relevance is smaller than max(relevance)*crop_th.
        Cropping is only applied, if receptive field 'rf' is set to True.
    kernel_size: scalar
        Parameter of the torchvision.transforms.functional.gaussian_blur function used to smooth the CRP heatmap.

    Returns:
    --------
    image: list of PIL.Image objects
        If 'rf' is True, reference images have different shapes.

    """

    if alpha > 1 or alpha < 0:
        raise ValueError("'alpha' must be between [0, 1]")
    if vis_th >= 1 or vis_th < 0:
        raise ValueError("'vis_th' must be between [0, 1)")
    if crop_th >= 1 or crop_th < 0:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []
    for i in range(len(data_batch)):

        img = data_batch[i]

        filtered_heat = gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0]
        filtered_heat = filtered_heat / filtered_heat.clamp(min=0).max()
        vis_mask = filtered_heat > vis_th

        if rf:
            row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)

            img_t = img[..., row1:row2, col1:col2]
            vis_mask_t = vis_mask[row1:row2, col1:col2]

            if img_t.sum() != 0 and vis_mask_t.sum() != 0:
                # check whether img_t or vis_mask_t is not empty
                img = img_t
                vis_mask = vis_mask_t

        inv_mask = ~vis_mask
        outside = (img * vis_mask).sum((1, 2)).mean(0) / stabilize(vis_mask.sum()) > 0.5

        img = img * vis_mask + img * inv_mask * alpha + outside * 0 * inv_mask * (1 - alpha)

        img = imgify(img.detach().cpu()).convert('RGBA')

        img_ = np.array(img).copy()
        img_[..., 3] = (vis_mask * 255).detach().cpu().numpy().astype(np.uint8)
        img_ = mystroke(Image.fromarray(img_), 1, color='black' if outside else 'black')

        img.paste(img_, (0, 0), img_)

        imgs.append(img.convert('RGB'))

    return imgs


def mystroke(img, size: int, color: str = 'black'):
    X, Y = img.size
    edge = img.filter(ImageFilter.FIND_EDGES).load()
    stroke = Image.new(img.mode, img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(stroke)
    fill = (0, 0, 0, 180) if color == 'black' else (255, 255, 255, 180)
    for x in range(X):
        for y in range(Y):
            if edge[x, y][3] > 0:
                draw.ellipse((x - size, y - size, x + size, y + size), fill=fill)
    stroke.paste(img, (0, 0), img)

    return stroke
