import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf
from typing import Dict, Tuple

from mile.utils.geometry_utils import get_out_of_view_mask
from mile.utils.instance_utils import convert_instance_mask_to_center_and_offset_label


class PreProcess(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.crop = tuple(cfg.IMAGE.CROP)
        self.route_map_size = cfg.ROUTE.SIZE

        self.bev_out_of_view_mask = get_out_of_view_mask(cfg)
        # Instance label parameters
        self.center_sigma = cfg.INSTANCE_SEG.CENTER_LABEL_SIGMA_PX
        self.ignore_index = cfg.INSTANCE_SEG.IGNORE_INDEX

        self.min_depth = cfg.BEV.FRUSTUM_POOL.D_BOUND[0]
        self.max_depth = cfg.BEV.FRUSTUM_POOL.D_BOUND[1]

        self.pixel_augmentation = PixelAugmentation(cfg)
        self.route_augmentation = RouteAugmentation(
                cfg.ROUTE.AUGMENTATION_DROPOUT,
                cfg.ROUTE.AUGMENTATION_END_OF_ROUTE,
                cfg.ROUTE.AUGMENTATION_SMALL_ROTATION,
                cfg.ROUTE.AUGMENTATION_LARGE_ROTATION,
                cfg.ROUTE.AUGMENTATION_DEGREES,
                cfg.ROUTE.AUGMENTATION_TRANSLATE,
                cfg.ROUTE.AUGMENTATION_SCALE,
                cfg.ROUTE.AUGMENTATION_SHEAR,
            )

        self.register_buffer('image_mean', torch.tensor(cfg.IMAGE.IMAGENET_MEAN).unsqueeze(1).unsqueeze(1))
        self.register_buffer('image_std', torch.tensor(cfg.IMAGE.IMAGENET_STD).unsqueeze(1).unsqueeze(1))

    def augmentation(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self.pixel_augmentation(batch)
        batch = self.route_augmentation(batch)
        return batch

    def prepare_bev_labels(self, batch):
        if 'birdview_label' in batch:
            # Mask bird's-eye view label pixels that are not visible from the input image
            batch['birdview_label'][:, :, :, self.bev_out_of_view_mask] = 0

            # Currently the frustum pooling is set up such that the bev features are rotated by 90 degrees clockwise
            batch['birdview_label'] = torch.rot90(batch['birdview_label'], k=-1, dims=[3, 4]).contiguous()

            # Compute labels at half, quarter, and 1/8th resolution
            batch['birdview_label_1'] = batch['birdview_label']
            h, w = batch['birdview_label'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'birdview_label_{downsample_factor}'] = functional_resize(
                    batch[f'birdview_label_{previous_label_factor}'], size, mode=tvf.InterpolationMode.NEAREST
                )

        if 'instance_label' in batch:
            # Mask elements not visible from the input image
            batch['instance_label'][:, :, :, self.bev_out_of_view_mask] = 0
            #  Currently the frustum pooling is set up such that the bev features are rotated by 90 degrees clockwise
            batch['instance_label'] = torch.rot90(batch['instance_label'], k=-1, dims=[3, 4]).contiguous()

            center_label, offset_label = convert_instance_mask_to_center_and_offset_label(
                batch['instance_label'], ignore_index=self.ignore_index, sigma=self.center_sigma,
            )
            batch['center_label'] = center_label
            batch['offset_label'] = offset_label

            # Compute labels at half, quarter, and 1/8th resolution
            batch['instance_label_1'] = batch['instance_label']
            batch['center_label_1'] = batch['center_label']
            batch['offset_label_1'] = batch['offset_label']

            h, w = batch['instance_label'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'instance_label_{downsample_factor}'] = functional_resize(
                    batch[f'instance_label_{previous_label_factor}'], size, mode=tvf.InterpolationMode.NEAREST
                )

                center_label, offset_label = convert_instance_mask_to_center_and_offset_label(
                    batch[f'instance_label_{downsample_factor}'], ignore_index=self.ignore_index,
                    sigma=self.center_sigma/downsample_factor,
                )
                batch[f'center_label_{downsample_factor}'] = center_label
                batch[f'offset_label_{downsample_factor}'] = offset_label

        if self.cfg.EVAL.RGB_SUPERVISION:
            # Compute labels at half, quarter, and 1/8th resolution
            batch['rgb_label_1'] = batch['image']
            h, w = batch['rgb_label_1'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'rgb_label_{downsample_factor}'] = functional_resize(
                    batch[f'rgb_label_{previous_label_factor}'],
                    size,
                    mode=tvf.InterpolationMode.BILINEAR,
                )

        return batch

    def forward(self, batch: Dict[str, torch.Tensor]):
        # Normalise from [0, 255] to [0, 1]
        batch['image'] = batch['image'].float() / 255

        if 'route_map' in batch:
            batch['route_map'] = batch['route_map'].float() / 255
            batch['route_map'] = functional_resize(batch['route_map'], size=(self.route_map_size, self.route_map_size))
        batch = functional_crop(batch, self.crop)
        if self.cfg.EVAL.RESOLUTION.ENABLED:
            batch = functional_resize_batch(batch, scale=1/self.cfg.EVAL.RESOLUTION.FACTOR)

        batch = self.prepare_bev_labels(batch)

        if self.training:
            batch = self.augmentation(batch)

        # Use imagenet mean and std normalisation, because we're loading pretrained backbones
        batch['image'] = (batch['image'] - self.image_mean) / self.image_std
        if 'route_map' in batch:
            batch['route_map'] = (batch['route_map'] - self.image_mean) / self.image_std

        if 'depth' in batch:
            batch['depth_mask'] = (batch['depth'] > self.min_depth) & (batch['depth'] < self.max_depth)

        return batch


def functional_crop(batch: Dict[str, torch.Tensor], crop: Tuple[int, int, int, int]):
    left, top, right, bottom = crop
    height = bottom - top
    width = right - left
    if 'image' in batch:
        batch['image'] = tvf.crop(batch['image'], top, left, height, width)
    if 'depth' in batch:
        batch['depth'] = tvf.crop(batch['depth'], top, left, height, width)
    if 'semseg' in batch:
        batch['semseg'] = tvf.crop(batch['semseg'], top, left, height, width)
    if 'intrinsics' in batch:
        intrinsics = batch['intrinsics'].clone()
        intrinsics[..., 0, 2] -= left
        intrinsics[..., 1, 2] -= top
        batch['intrinsics'] = intrinsics

    return batch


def functional_resize_batch(batch, scale):
    b, s, c, h, w = batch['image'].shape
    h1, w1 = int(round(h * scale)), int(round(w * scale))
    size = (h1, w1)
    if 'image' in batch:
        image = batch['image'].view(b*s, c, h, w)
        image = tvf.resize(image, size, antialias=True)
        batch['image'] = image.view(b, s, c, h1, w1)
    if 'intrinsics' in batch:
        intrinsics = batch['intrinsics'].clone()
        intrinsics[..., :2, :] *= scale
        batch['intrinsics'] = intrinsics

    return batch


def functional_resize(x, size, mode=tvf.InterpolationMode.NEAREST):
    b, s, c, h, w = x.shape
    x = x.view(b * s, c, h, w)
    x = tvf.resize(x, size, interpolation=mode)
    x = x.view(b, s, c, *size)

    return x


class PixelAugmentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO replace with ImageApply([RandomBlurSharpen(), RandomColorJitter(), ...])
        self.blur_prob = cfg.IMAGE.AUGMENTATION.BLUR_PROB
        self.sharpen_prob = cfg.IMAGE.AUGMENTATION.SHARPEN_PROB
        self.blur_window = cfg.IMAGE.AUGMENTATION.BLUR_WINDOW
        self.blur_std = cfg.IMAGE.AUGMENTATION.BLUR_STD
        self.sharpen_factor = cfg.IMAGE.AUGMENTATION.SHARPEN_FACTOR
        assert self.blur_prob + self.sharpen_prob <= 1

        self.color_jitter = transforms.RandomApply(nn.ModuleList([
            transforms.ColorJitter(
                cfg.IMAGE.AUGMENTATION.COLOR_JITTER_BRIGHTNESS,
                cfg.IMAGE.AUGMENTATION.COLOR_JITTER_CONTRAST,
                cfg.IMAGE.AUGMENTATION.COLOR_JITTER_SATURATION,
                cfg.IMAGE.AUGMENTATION.COLOR_JITTER_HUE
            )
        ]), cfg.IMAGE.AUGMENTATION.COLOR_PROB)

    def forward(self, batch: Dict[str, torch.Tensor]):
        image = batch['image']
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # random blur
                rand_value = torch.rand(1)
                if rand_value < self.blur_prob:
                    std = torch.empty(1).uniform_(self.blur_std[0], self.blur_std[1]).item()
                    image[i, j] = tvf.gaussian_blur(image[i, j], self.blur_window, std)
                # random sharpen
                elif rand_value < self.blur_prob + self.sharpen_prob:
                    factor = torch.empty(1).uniform_(self.sharpen_factor[0], self.sharpen_factor[1]).item()
                    image[i, j] = tvf.adjust_sharpness(image[i, j], factor)

                # random color jitter
                image[i, j] = self.color_jitter(image[i, j])

        batch['image'] = image
        return batch


class RouteAugmentation(nn.Module):
    def __init__(self, drop=0.025, end_of_route=0.025, small_rotation=0.025, large_rotation=0.025, degrees=8.0,
                 translate=(.1, .1), scale=(.95, 1.05), shear=(.1, .1)):
        super().__init__()
        assert drop + end_of_route + small_rotation + large_rotation <= 1
        self.drop = drop  # random dropout of map
        self.end_of_route = end_of_route  # probability of end of route augmentation
        self.small_rotation = small_rotation  # probability of doing small rotation
        self.large_rotation = large_rotation  # probability of doing large rotation (arbitrary orientation)
        self.small_perturbation = transforms.RandomAffine(degrees, translate, scale, shear)  # small rotation
        self.large_perturbation = transforms.RandomAffine(180, translate, scale, shear)  # arbitrary orientation

    def forward(self, batch):
        if 'route_map' in batch:
            route_map = batch['route_map']

            # TODO: make augmentation independent of the sequence dimension?
            for i in range(route_map.shape[0]):
                rand_value = torch.rand(1)
                if rand_value < self.drop:
                    route_map[i] = torch.zeros_like(route_map[i])
                elif rand_value < self.drop + self.end_of_route:
                    height = torch.randint(route_map[i].shape[-2], (1,))
                    route_map[i][:, :, :height] = 0
                elif rand_value < self.drop + self.end_of_route + self.small_rotation:
                    route_map[i] = self.small_perturbation(route_map[i])
                elif rand_value < self.drop + self.end_of_route + self.small_rotation + self.large_rotation:
                    route_map[i] = self.large_perturbation(route_map[i])

            batch['route_map'] = route_map

        return batch
