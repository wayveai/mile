import os
from glob import glob
from PIL import Image

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader

from mile.constants import CARLA_FPS
from mile.data.dataset_utils import integer_to_binary, calculate_birdview_labels
from mile.utils.geometry_utils import get_out_of_view_mask


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.BATCHSIZE
        self.sequence_length = self.cfg.RECEPTIVE_FIELD + self.cfg.FUTURE_HORIZON

        # Will be populated with self.setup()
        self.train_dataset, self.val_dataset = None, None

    def setup(self, stage=None):
        self.train_dataset = CarlaDataset(self.cfg, mode='train', sequence_length=self.sequence_length)
        self.val_dataset = CarlaDataset(self.cfg, mode='val', sequence_length=self.sequence_length)

        print(f'{len(self.train_dataset)} data points in {self.train_dataset.dataset_path}')
        print(f'{len(self.val_dataset)} data points in {self.val_dataset.dataset_path}')

        self.train_sampler = None
        self.val_sampler = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_sampler is None,
            num_workers=self.cfg.N_WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.N_WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=self.val_sampler,
        )


class CarlaDataset(Dataset):
    def __init__(self, cfg, mode='train', sequence_length=1):
        self.cfg = cfg
        self.mode = mode
        self.sequence_length = sequence_length

        self.dataset_path = os.path.join(self.cfg.DATASET.DATAROOT, self.cfg.DATASET.VERSION, mode)
        self.intrinsics, self.extrinsics = calculate_geometry(self.cfg)
        self.bev_out_of_view_mask = get_out_of_view_mask(self.cfg)

        # Iterate over all runs in the data folder

        self.data = dict()

        towns = sorted(glob(os.path.join(self.dataset_path, '*')))
        for town_path in towns:
            town = os.path.basename(town_path)

            runs = sorted(glob(os.path.join(self.dataset_path, town, '*')))
            for run_path in runs:
                run = os.path.basename(run_path)
                pd_dataframe_path = os.path.join(run_path, 'pd_dataframe.pkl')

                if os.path.isfile(pd_dataframe_path):
                    self.data[f'{town}/{run}'] = pd.read_pickle(pd_dataframe_path)

        self.data_pointers = self.get_data_pointers()

    def get_data_pointers(self):
        data_pointers = []

        n_filtered_run = 0
        for run, data_run in self.data.items():
            # Calculate normalised reward of the run
            run_length = len(data_run['reward'])
            cumulative_reward = data_run['reward'].sum()
            normalised_reward = cumulative_reward / run_length
            if normalised_reward < self.cfg.DATASET.FILTER_NORM_REWARD:
                n_filtered_run += 1
                continue

            stride = int(self.cfg.DATASET.STRIDE_SEC * CARLA_FPS)
            # Loop across all elements in the dataset, and make all elements in a sequence belong to the same run
            start_index = int(CARLA_FPS * self.cfg.DATASET.FILTER_BEGINNING_OF_RUN_SEC)
            total_length = len(data_run) - stride * self.sequence_length
            for i in range(start_index, total_length):
                frame_indices = range(i, i + stride * self.sequence_length, stride)
                data_pointers.append((run, list(frame_indices)))

        print(f'Filtered {n_filtered_run} runs in {self.dataset_path}')

        if self.cfg.EVAL.DATASET_REDUCTION:
            import random
            random.seed(0)
            final_size = int(len(data_pointers) / self.cfg.EVAL.DATASET_REDUCTION_FACTOR)
            data_pointers = random.sample(data_pointers, final_size)

        return data_pointers

    def __len__(self):
        return len(self.data_pointers)

    def __getitem__(self, i):
        batch = {}

        run_id, indices = self.data_pointers[i]
        for t in indices:
            single_element_t = self.load_single_element_time_t(run_id, t)

            for k, v in single_element_t.items():
                batch[k] = batch.get(k, []) + [v]

        for k, v in batch.items():
            batch[k] = torch.from_numpy(np.stack(v))

        return batch

    def load_single_element_time_t(self, run_id, t):
        data_row = self.data[run_id].iloc[t]
        single_element_t = {}

        # Load image
        image = Image.open(
            os.path.join(self.dataset_path, run_id, data_row['image_path'])
        )
        image = np.asarray(image).transpose((2, 0, 1))
        single_element_t['image'] = image

        # Load route map
        route_map = Image.open(
            os.path.join(self.dataset_path, run_id, data_row['routemap_path'])
        )
        route_map = np.asarray(route_map)[None]
        # Make the grayscale image an RGB image
        _, h, w = route_map.shape
        route_map = np.broadcast_to(route_map, (3, h, w)).copy()
        single_element_t['route_map'] = route_map

        # Load bird's-eye view segmentation label
        birdview = np.asarray(Image.open(
            os.path.join(self.dataset_path, run_id, data_row['birdview_path'])
        ))
        h, w = birdview.shape
        n_classes = data_row['n_classes']
        birdview = integer_to_binary(birdview.reshape(-1), n_classes).reshape(h, w, n_classes)
        birdview = birdview.transpose((2, 0, 1))
        single_element_t['birdview'] = birdview
        birdview_label = calculate_birdview_labels(torch.from_numpy(birdview), n_classes).numpy()
        birdview_label = birdview_label[None]
        single_element_t['birdview_label'] = birdview_label

        # TODO: get person and car instance ids with json
        instance_mask = birdview[3].astype(np.bool) | birdview[4].astype(np.bool)
        instance_label, _ = scipy.ndimage.label(instance_mask[None].astype(np.int64))
        single_element_t['instance_label'] = instance_label

        # Load action and reward
        throttle, steering, brake = data_row['action']
        throttle_brake = throttle if throttle > 0 else -brake

        single_element_t['steering'] = np.array([steering], dtype=np.float32)
        single_element_t['throttle_brake'] = np.array([throttle_brake], dtype=np.float32)
        single_element_t['speed'] = data_row['speed']

        single_element_t['reward'] = np.array([data_row['reward']], dtype=np.float32).clip(-1.0, 1.0)
        single_element_t['value_function'] = np.array([data_row['value']], dtype=np.float32)

        # Geometry
        single_element_t['intrinsics'] = self.intrinsics.copy()
        single_element_t['extrinsics'] = self.extrinsics.copy()

        return single_element_t


def calculate_geometry(cfg):
    """ Intrinsics and extrinsics for a single camera.
    See https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/leaderboard/camera.py
    and https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/recording/sensors/camera.py
    """

    # Intrinsics
    fov = cfg.IMAGE.FOV
    h, w = cfg.IMAGE.SIZE

    f = w / (2 * np.tan(fov * np.pi / 360.0))
    cx = w / 2
    cy = h / 2
    intrinsics = np.float32([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]])

    # Extrinsics
    forward, right, up = cfg.IMAGE.CAMERA_POSITION
    pitch, yaw, roll = cfg.IMAGE.CAMERA_ROTATION

    extrinsics = get_extrinsics(forward, right, up, pitch, yaw, roll)
    return intrinsics, extrinsics


def get_extrinsics(forward, right, up, pitch, yaw, roll):
    # After multiplying the image coordinates by in the inverse intrinsics,
    # the resulting coordinates are defined with the axes (right, down, forward)
    assert pitch == yaw == roll == 0.0

    # After multiplying by the extrinsics, we want the axis to be (forward, left, up), and centered in the
    # inertial center of the ego-vehicle.
    mat = np.float32([
        [0,  0,  1, forward],
        [-1, 0,  0, -right],
        [0,  -1, 0, up],
        [0,  0,  0, 1],
    ])

    return mat
