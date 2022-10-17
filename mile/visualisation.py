import os
import matplotlib.pylab
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision.transforms.functional as tvf

from mile.constants import EGO_VEHICLE_DIMENSION, BIRDVIEW_COLOURS


DEFAULT_COLORMAP = matplotlib.pylab.cm.jet
HEATMAP_PALETTE = (
    torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]).permute(1, 0).view(1, 3, 4, 1)
)


def prepare_final_display_image(img_rgb, route_map, birdview_label, birdview_pred, render_dict, is_predicting=False,
                                future_sec=None):
    if is_predicting:
        pred_colour = [79, 171, 198]
    else:
        pred_colour = [0, 0, 0]

    rgb_height, rgb_width = img_rgb.shape[:2]
    birdview_height, birdview_width = birdview_pred.shape[:2]

    final_display_image_margin_height = 110
    final_display_image_margin_width = 0
    final_height = max(rgb_height, birdview_height) + final_display_image_margin_height
    final_width = rgb_width + 2 * birdview_width + final_display_image_margin_width
    final_display_image = 255 * np.ones([final_height, final_width, 3], dtype=np.uint8)
    final_display_image[:rgb_height, :rgb_width] = img_rgb

    #  add route map
    route_map_height, route_map_width = route_map.shape[:2]
    margin = 10
    route_map_width_slice = slice(margin, route_map_height + margin)
    route_map_height_slice = slice(rgb_width - route_map_width - margin, rgb_width - margin)
    final_display_image[route_map_width_slice, route_map_height_slice] = \
        (0.3 * final_display_image[route_map_width_slice, route_map_height_slice]
         + 0.7 * route_map
         ).astype(np.uint8)

    # Bev prediction
    final_display_image[:birdview_height, rgb_width:(rgb_width + birdview_width)] = birdview_label
    final_display_image[:birdview_height, (rgb_width + birdview_width):(rgb_width + 2 * birdview_width)] = birdview_pred

    # Action gauges
    final_display_image = add_action_gauges(
        final_display_image, render_dict, height=birdview_height + 45, width=rgb_width, pred_colour=pred_colour
    )

    # Legend
    final_display_image = add_legend(final_display_image, f'RGB input (time t)',
                                     (0, rgb_height + 5), colour=[0, 0, 0], size=24)
    final_display_image = add_legend(final_display_image, f'Ground truth BEV (time t)',
                                     (rgb_width, birdview_height + 5), colour=[0, 0, 0], size=24)
    label = 'Pred. BEV (time t)'
    if future_sec is not None:
        label = f'Pred. BEV (time t + {future_sec:.1f}s)'
    final_display_image = add_legend(final_display_image, label,
                                     (rgb_width + birdview_width, birdview_height + 5), colour=pred_colour, size=24)
    if is_predicting:
        final_display_image = add_legend(final_display_image, 'IMAGINING',
                                         (rgb_width + birdview_width + 5, birdview_height - 30), colour=pred_colour,
                                         size=24)
    return final_display_image


def upsample_bev(x, size=(320, 320)):
    _, h, w = x.shape
    x = tvf.resize(
        x.unsqueeze(0), size, interpolation=tvf.InterpolationMode.NEAREST,
    )
    return x[0]


def convert_bev_to_image(bev, cfg, upsample_factor=2):
    bev = BIRDVIEW_COLOURS[bev]
    bev_pixel_per_m = upsample_factor*int(1 / cfg.BEV.RESOLUTION)
    ego_vehicle_bottom_offset_pixel = int(cfg.BEV.SIZE[0] / 2 + cfg.BEV.OFFSET_FORWARD)
    bev = add_ego_vehicle(
        bev,
        pixel_per_m=bev_pixel_per_m,
        ego_vehicle_bottom_offset_pixel=ego_vehicle_bottom_offset_pixel,
    )
    bev = make_contour(bev, colour=[0, 0, 0])
    return bev


def add_ego_vehicle(img, pixel_per_m=5, ego_vehicle_bottom_offset_pixel=32):
    h, w = img.shape[:2]
    # Assume vehicle is symmetrical in the x and y axis.
    ego_vehicle_dimension_pixel = [int((x/2)*pixel_per_m) for x in EGO_VEHICLE_DIMENSION]

    bottom_coordinate = h - ego_vehicle_bottom_offset_pixel - ego_vehicle_dimension_pixel[0]
    top_coordinate = h - ego_vehicle_bottom_offset_pixel + ego_vehicle_dimension_pixel[0] + 1
    left_coordinate = w//2 - ego_vehicle_dimension_pixel[1]
    right_coordinate = w//2 + ego_vehicle_dimension_pixel[1] + 1

    copy_img = img.copy()
    copy_img[bottom_coordinate:top_coordinate, left_coordinate:right_coordinate] = [0, 0, 0]
    return copy_img


def make_contour(img, colour=[0, 0, 0], double_line=False):
    h, w = img.shape[:2]
    out = img.copy()
    # Vertical lines
    out[np.arange(h), np.repeat(0, h)] = colour
    out[np.arange(h), np.repeat(w - 1, h)] = colour

    # Horizontal lines
    out[np.repeat(0, w), np.arange(w)] = colour
    out[np.repeat(h - 1, w), np.arange(w)] = colour

    if double_line:
        out[np.arange(h), np.repeat(1, h)] = colour
        out[np.arange(h), np.repeat(w - 2, h)] = colour

        # Horizontal lines
        out[np.repeat(1, w), np.arange(w)] = colour
        out[np.repeat(h - 2, w), np.arange(w)] = colour
    return out


def merge_sparse_image_to_image_torch(
    base_image: torch.Tensor,
    sparse_image: torch.Tensor,
    transparency: float = 0.4,
) -> torch.Tensor:
    assert base_image.shape == sparse_image.shape
    canvas = base_image.clone()
    mask = (sparse_image > 0).any(dim=0, keepdim=True).expand(*base_image.shape)

    canvas[mask] = ((1 - transparency) * sparse_image[mask] + transparency * base_image[mask]).to(torch.uint8)
    return canvas


def add_legend(img, text='hello', position=(0, 0), colour=[255, 255, 255], size=14):
    font_path = 'DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, size)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, tuple(colour), font=font)
    return np.array(pil_img)


def add_action_gauges(img, render_dict, height, width, pred_colour):
    def plot_gauge(img, label, value, gauge_height, color=(79, 171, 198), max_value=None):
        bar_height = 15
        bar_width = 150
        centering_offset = 40
        width_offset = bar_width + width + 200 + centering_offset
        cursor = value
        if max_value is not None:
            cursor /= max_value
        if cursor > 0:
            start = 0
            end = int(cursor * bar_width)
        else:
            start = int(cursor * bar_width)
            end = 0

        # fill
        img[gauge_height:gauge_height + bar_height, width_offset + start:width_offset + end] = color
        # contour
        height_slice = slice(gauge_height, gauge_height + bar_height)
        width_slice = slice(width_offset - bar_width, width_offset + bar_width)
        img[height_slice, width_slice] = make_contour(img[height_slice, width_slice], colour=[0, 0, 0])

        # Middle gauge
        img[gauge_height - 2:gauge_height + bar_height + 2, width_offset:width_offset + 1] = (0, 0, 0)
        # Add labels
        img = add_legend(img, f'{label}:', (width + centering_offset - 35, gauge_height - bar_height // 2), pred_colour,
                         size=24)
        img = add_legend(img, f'{value:.2f}', (width_offset + bar_width + 10, gauge_height - bar_height // 2),
                         pred_colour,
                         size=24)
        return img

    acceleration = render_dict['throttle_brake'].item()
    steering = render_dict['steering'].item()

    img = plot_gauge(img, 'Pred. acceleration', acceleration, gauge_height=height + 10, color=(224, 102, 102))
    img = plot_gauge(img, 'Pred. steering', steering, gauge_height=height + 40, color=(255, 127, 80))
    return img


def heatmap_image(
        image: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap = DEFAULT_COLORMAP, autoscale: bool = True
) -> np.ndarray:
    """Colorize an 1 or 2 channel image with a colourmap."""
    if not issubclass(image.dtype.type, np.floating):
        raise ValueError(f"Expected a ndarray of float type, but got dtype {image.dtype}")
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[0] in [1, 2])):
        raise ValueError(f"Expected a ndarray of shape [H, W] or [1, H, W] or [2, H, W], but got shape {image.shape}")
    heatmap_np = apply_colour_map(image.copy(), cmap=cmap, autoscale=autoscale)
    heatmap_np = np.uint8(heatmap_np * 255)
    return heatmap_np


def apply_colour_map(
        image: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap = DEFAULT_COLORMAP, autoscale: bool = False
) -> np.ndarray:
    """
    Applies a colour map to the given 1 or 2 channel numpy image. if 2 channel, must be 2xHxW.
    Returns a HxWx3 numpy image
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
        if image.ndim == 3:
            image = image[0]
        # grayscale scalar image
        if autoscale:
            image = _normalise(image)
        return cmap(image)[:, :, :3]
    if image.shape[0] == 2:
        # 2 dimensional UV
        return flow_to_image(image, autoscale=autoscale)
    if image.shape[0] == 3:
        # normalise rgb channels
        if autoscale:
            image = _normalise(image)
        return np.transpose(image, axes=[1, 2, 0])
    raise Exception('Image must be 1, 2 or 3 channel to convert to colour_map (CxHxW)')


def _normalise(image: np.ndarray) -> np.ndarray:
    lower = np.min(image)
    delta = np.max(image) - lower
    if delta == 0:
        delta = 1
    image = (image.astype(np.float32) - lower) / delta
    return image


def flow_to_image(flow: np.ndarray, autoscale: bool = False) -> np.ndarray:
    """
    Applies colour map to flow which should be a 2 channel image tensor HxWx2. Returns a HxWx3 numpy image
    Code adapted from: https://github.com/liruoteng/FlowNet/blob/master/models/flownet/scripts/flowlib.py
    """
    u = flow[0, :, :]
    v = flow[1, :, :]

    # Convert to polar coordinates
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = np.max(rad)

    # Normalise flow maps
    if autoscale:
        u /= maxrad + np.finfo(float).eps
        v /= maxrad + np.finfo(float).eps

    # visualise flow with cmap
    return np.uint8(compute_color(u, v) * 255)


def compute_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert u.shape == v.shape
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nan_mask = np.isnan(u) | np.isnan(v)
    u[nan_mask] = 0
    v[nan_mask] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    f_k = (a + 1) / 2 * (ncols - 1) + 1
    k_0 = np.floor(f_k).astype(int)
    k_1 = k_0 + 1
    k_1[k_1 == ncols + 1] = 1
    f = f_k - k_0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k_0 - 1] / 255
        col1 = tmp[k_1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = col * (1 - nan_mask)

    return img


def make_color_wheel() -> np.ndarray:
    """
    Create colour wheel.
    Code adapted from https://github.com/liruoteng/FlowNet/blob/master/models/flownet/scripts/flowlib.py
    """
    red_yellow = 15
    yellow_green = 6
    green_cyan = 4
    cyan_blue = 11
    blue_magenta = 13
    magenta_red = 6

    ncols = red_yellow + yellow_green + green_cyan + cyan_blue + blue_magenta + magenta_red
    colorwheel = np.zeros([ncols, 3])

    col = 0

    # red_yellow
    colorwheel[0:red_yellow, 0] = 255
    colorwheel[0:red_yellow, 1] = np.transpose(np.floor(255 * np.arange(0, red_yellow) / red_yellow))
    col += red_yellow

    # yellow_green
    colorwheel[col: col + yellow_green, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, yellow_green) / yellow_green)
    )
    colorwheel[col: col + yellow_green, 1] = 255
    col += yellow_green

    # green_cyan
    colorwheel[col: col + green_cyan, 1] = 255
    colorwheel[col: col + green_cyan, 2] = np.transpose(np.floor(255 * np.arange(0, green_cyan) / green_cyan))
    col += green_cyan

    # cyan_blue
    colorwheel[col: col + cyan_blue, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, cyan_blue) / cyan_blue))
    colorwheel[col: col + cyan_blue, 2] = 255
    col += cyan_blue

    # blue_magenta
    colorwheel[col: col + blue_magenta, 2] = 255
    colorwheel[col: col + blue_magenta, 0] = np.transpose(np.floor(255 * np.arange(0, blue_magenta) / blue_magenta))
    col += +blue_magenta

    # magenta_red
    colorwheel[col: col + magenta_red, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, magenta_red) / magenta_red))
    colorwheel[col: col + magenta_red, 0] = 255

    return colorwheel
