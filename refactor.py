import gym
import json
from pathlib import Path
import sys
import cv2

from gym.wrappers.monitoring.video_recorder import ImageEncoder
from stable_baselines3.common.vec_env.base_vec_env import tile_images

import carla
from carla_gym.utils import config_utils
from utils import server_utils
from mile.constants import CARLA_FPS
import time

import logging
from collections import deque

import torch
from torchmetrics import JaccardIndex

from carla_gym.utils.config_utils import load_entry_point
from mile.constants import CARLA_FPS, DISPLAY_SEGMENTATION
from mile.data.dataset import calculate_geometry_from_config
from mile.data.dataset_utils import preprocess_birdview_and_routemap, preprocess_measurements, calculate_birdview_labels
from mile.trainer import WorldModelTrainer

"""Adapted from https://github.com/zhejz/carla-roach CC-BY-NC 4.0 license."""
import numpy as np
import torch
import carla

from mile.visualisation import upsample_bev, add_legend, add_ego_vehicle, make_contour
from mile.constants import EGO_VEHICLE_DIMENSION, BIRDVIEW_COLOURS


obs_configs = {
    'hero': {'speed': {'module': 'actor_state.speed'}, 'gnss': {'module': 'navigation.gnss'},
             'central_rgb': {'module': 'camera.rgb', 'fov': 100, 'width': 960, 'height': 600,
                             'location': [-1.5, 0.0, 2.0], 'rotation': [0.0, 0.0, 0.0]},
             'route_plan': {'module': 'navigation.waypoint_plan', 'steps': 20},
             'birdview': {'module': 'birdview.chauffeurnet', 'width_in_pixels': 192,
                          'pixels_ev_to_bottom': 32, 'pixels_per_meter': 5.0,
                          'history_idx': [-16, -11, -6, -1], 'scale_bbox': True, 'scale_mask_col': 1.0}}
}

reward_configs = {'hero': {'entry_point': 'reward.valeo_action:ValeoAction'}}
terminal_configs = {'hero': {'entry_point': 'terminal.leaderboard:Leaderboard'}}
env_configs = {'carla_map': 'Town02', 'routes_group': None, 'weather_group': 'new'}

agent_obs_configs = {
    'speed': {'module': 'actor_state.speed'},
    'gnss': {'module': 'navigation.gnss'},
    'central_rgb': {'module': 'camera.rgb', 'fov': 100, 'width': 960, 'height': 600,
                    'location': [-1.5, 0.0, 2.0], 'rotation': [0.0, 0.0, 0.0]},
    'route_plan': {'module': 'navigation.waypoint_plan', 'steps': 20},
    'birdview': {'module': 'birdview.chauffeurnet',
                 'width_in_pixels': 192,
                 'pixels_ev_to_bottom': 32,
                 'pixels_per_meter': 5.0,
                 'history_idx': [-16, -11, -6, -1],
                 'scale_bbox': True,
                 'scale_mask_col': 1.0}
}

test_suites = [
    {'env_id': 'LeaderBoard-v0', 'env_configs': {'carla_map': 'Town02', 'routes_group': None, 'weather_group': 'new'}},
    {'env_id': 'LeaderBoard-v0', 'env_configs': {'carla_map': 'Town05', 'routes_group': None, 'weather_group': 'new'}}
]

policy_cfg = {
    'BEV': {
        'RESOLUTION': 0.2,
        'SIZE': [192, 192],
        'OFFSET_FORWARD': -64
    },
    'IMAGE': {
        'IMAGENET_MEAN': (0.485, 0.456, 0.406),
        'IMAGENET_STD': (0.229, 0.224, 0.225)
    }
}

def main():
    server_utils.kill_carla(2000)
    server_manager = server_utils.CarlaServerManager('/home/carla/CarlaUE4.sh', port=2000)
    server_manager.start()

    env = gym.make(
        'LeaderBoard-v0',
        obs_configs=obs_configs,
        reward_configs=reward_configs,
        terminal_configs=terminal_configs,
        host='localhost',
        port=2000,
        seed=2021,
        no_rendering=False,
        **env_configs)

    task_idx = 0
    env.set_task_idx(task_idx)
    agent = MileAgent(
        agent_obs_configs, '/home/carla/mile.ckpt'
    )
    actor_id = 'hero'

    obs = env.reset()
    for counter in range(1000):
        print(counter)
        control_dict = {}
        control_dict[actor_id] = carla.VehicleControl(
            throttle=0.843839, steer=-0.001, brake=0.000000)
        policy_input = agent.run_step(obs[actor_id])

        img = im_render(policy_input, policy_cfg)
        cv2.imshow('a', img)
        cv2.waitKey(1)

        obs, reward, done, info = env.step(control_dict)


    env.close()
    server_manager.stop()


class MileAgent:
    def __init__(self, obs_configs, cfg_ckpt):
        self._render_dict = None

        self._obs_configs = obs_configs
        # for debug view
        self._obs_configs['route_plan'] = {'module': 'navigation.waypoint_plan', 'steps': 20}

        trainer = WorldModelTrainer.load_from_checkpoint(cfg_ckpt, pretrained_path=cfg_ckpt)
        print(f'Loading world model weights from {cfg_ckpt}')
        self._policy = trainer.to('cuda')
        game_frequency = CARLA_FPS
        model_stride_sec = self._policy.cfg.DATASET.STRIDE_SEC
        receptive_field = trainer.model.receptive_field
        n_image_per_stride = int(game_frequency * model_stride_sec)

        self.input_buffer_size = (receptive_field - 1) * n_image_per_stride + 1
        self.sequence_indices = range(0, self.input_buffer_size, n_image_per_stride)

        self._policy = self._policy.eval()

        self.policy_input_queue = {
            'image': deque(maxlen=self.input_buffer_size),
            'route_map': deque(maxlen=self.input_buffer_size),
            'route_command': deque(maxlen=self.input_buffer_size),
            'gps_vector': deque(maxlen=self.input_buffer_size),
            'route_command_next': deque(maxlen=self.input_buffer_size),
            'gps_vector_next': deque(maxlen=self.input_buffer_size),
            'speed': deque(maxlen=self.input_buffer_size),
            'intrinsics': deque(maxlen=self.input_buffer_size),
            'extrinsics': deque(maxlen=self.input_buffer_size),
            'birdview': deque(maxlen=self.input_buffer_size),
            'birdview_label': deque(maxlen=self.input_buffer_size),
        }
        self.action_queue = deque(maxlen=self.input_buffer_size)

        # Custom metrics
        if self._policy.cfg.SEMANTIC_SEG.ENABLED and DISPLAY_SEGMENTATION:
            self.iou = JaccardIndex(task='multiclass', num_classes=self._policy.cfg.SEMANTIC_SEG.N_CHANNELS).cuda()
            self.real_time_iou = JaccardIndex(
                task='multiclass', num_classes=self._policy.cfg.SEMANTIC_SEG.N_CHANNELS, compute_on_step=True,
            ).cuda()

        # logger
        for v in self.policy_input_queue.values():
            v.clear()

        self.action_queue.clear()

    def run_step(self, input_data):
        policy_input = self.preprocess_data(input_data)
        # Forward pass
        with torch.no_grad():
            self._policy.preprocess(policy_input)
            # output = self._policy(policy_input, deployment=True)
        return policy_input

    def preprocess_data(self, input_data):
        # Fill the input queue with new elements
        image = input_data['central_rgb']['data'].transpose((2, 0, 1))

        route_command, gps_vector = preprocess_measurements(
            input_data['gnss']['command'].squeeze(0),
            input_data['gnss']['gnss'],
            input_data['gnss']['target_gps'],
            input_data['gnss']['imu'],
        )

        route_command_next, gps_vector_next = preprocess_measurements(
            input_data['gnss']['command_next'].squeeze(0),
            input_data['gnss']['gnss'],
            input_data['gnss']['target_gps_next'],
            input_data['gnss']['imu'],
        )

        birdview, route_map = preprocess_birdview_and_routemap(torch.from_numpy(input_data['birdview']['masks']).cuda())
        birdview_label = calculate_birdview_labels(birdview, birdview.shape[0]).unsqueeze(0)

        # Make route_map an RGB image
        route_map = route_map.unsqueeze(0).expand(3, -1, -1)
        speed = input_data['speed']['forward_speed']
        intrinsics, extrinsics = calculate_geometry_from_config(self._policy.cfg)

        # Using gpu inputs
        self.policy_input_queue['image'].append(torch.from_numpy(image.copy()).cuda())
        self.policy_input_queue['route_command'].append(torch.from_numpy(route_command).cuda())
        self.policy_input_queue['gps_vector'].append(torch.from_numpy(gps_vector).cuda())
        self.policy_input_queue['route_command_next'].append(torch.from_numpy(route_command_next).cuda())
        self.policy_input_queue['gps_vector_next'].append(torch.from_numpy(gps_vector_next).cuda())
        self.policy_input_queue['route_map'].append(route_map)
        self.policy_input_queue['speed'].append(torch.from_numpy(speed).cuda())
        self.policy_input_queue['intrinsics'].append(torch.from_numpy(intrinsics).cuda())
        self.policy_input_queue['extrinsics'].append(torch.from_numpy(extrinsics).cuda())

        self.policy_input_queue['birdview'].append(birdview)
        self.policy_input_queue['birdview_label'].append(birdview_label)

        for key in self.policy_input_queue.keys():
            while len(self.policy_input_queue[key]) < self.input_buffer_size:
                self.policy_input_queue[key].append(self.policy_input_queue[key][-1])

        if len(self.action_queue) == 0:
            self.action_queue.append(torch.zeros(2, device=torch.device('cuda')))
        while len(self.action_queue) < self.input_buffer_size:
            self.action_queue.append(self.action_queue[-1])

        # Prepare model input
        policy_input = {}
        for key in self.policy_input_queue.keys():
            policy_input[key] = torch.stack(list(self.policy_input_queue[key]), axis=0).unsqueeze(0).clone()

        action_input = torch.stack(list(self.action_queue), axis=0).unsqueeze(0).float()

        # We do not have access to the last action, as it is the one we're going to compute.
        action_input = torch.cat([action_input[:, 1:], torch.zeros_like(action_input[:, -1:])], dim=1)
        policy_input['action'] = action_input

        # Select right elements in the sequence
        for k, v in policy_input.items():
            policy_input[k] = v[:, self.sequence_indices]

        return policy_input


def im_render(policy_input, policy_cfg, upsample_bev_factor=2):
    im_rgb = policy_input['image'][0, -1].cpu().numpy().transpose((1, 2, 0))
    route_map = policy_input['route_map'][0, -1].cpu().numpy().transpose((1, 2, 0))

    # Un-normalise images
    img_mean = np.array(policy_cfg['IMAGE']['IMAGENET_MEAN'])
    img_std = np.array(policy_cfg['IMAGE']['IMAGENET_STD'])
    im_rgb = (255 * (im_rgb * img_std + img_mean)).astype(np.uint8)
    route_map = (255 * (route_map * img_std + img_mean)).astype(np.uint8)

    birdview_label = policy_input['birdview_label'][0, -1]
    birdview_label = torch.rot90(birdview_label, k=1, dims=[1, 2])
    birdview_label = upsample_bev(birdview_label)
    # Add colours
    birdview_label_rendered = convert_bev_to_image(
        birdview_label.cpu().numpy()[0], bev_cfg=policy_cfg['BEV'], upsample_factor=upsample_bev_factor,
    )

    final_display_image = prepare_final_display_image(
        im_rgb, route_map, birdview_label_rendered
    )

    return final_display_image


def convert_bev_to_image(bev, bev_cfg, upsample_factor=2):
    bev = BIRDVIEW_COLOURS[bev]
    bev_pixel_per_m = upsample_factor*int(1 / bev_cfg['RESOLUTION'])
    ego_vehicle_bottom_offset_pixel = int(bev_cfg['SIZE'][0] / 2 + bev_cfg['OFFSET_FORWARD'])
    bev = add_ego_vehicle(
        bev,
        pixel_per_m=bev_pixel_per_m,
        ego_vehicle_bottom_offset_pixel=ego_vehicle_bottom_offset_pixel,
    )
    bev = make_contour(bev, colour=[0, 0, 0])
    return bev


def prepare_final_display_image(img_rgb, route_map, birdview_label):
    pred_colour = [0, 0, 0]

    rgb_height, rgb_width = img_rgb.shape[:2]
    birdview_height, birdview_width = birdview_label.shape[:2]

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

    # Legend
    final_display_image = add_legend(final_display_image, f'RGB input (time t)',
                                     (0, rgb_height + 5), colour=[0, 0, 0], size=24)
    final_display_image = add_legend(final_display_image, f'Ground truth BEV (time t)',
                                     (rgb_width, birdview_height + 5), colour=[0, 0, 0], size=24)
    label = 'Pred. BEV (time t)'
    final_display_image = add_legend(final_display_image, label,
                                     (rgb_width + birdview_width, birdview_height + 5), colour=pred_colour, size=24)
    return final_display_image


"""
BATCHSIZE: 8

DATASET:
  DATAROOT: /mnt/local/datasets/paper_dataset
  FILTER_BEGINNING_OF_RUN_SEC: 1.0
  FILTER_NORM_REWARD: 0.6
  FREQUENCY: 25
  STRIDE_SEC: 0.2
  VERSION: trainval
EVAL:
  CHECKPOINT_PATH: 
  DATASET_REDUCTION: False
  DATASET_REDUCTION_FACTOR: 1
  NO_LIFTING: False
  RESOLUTION:
    ENABLED: False
    FACTOR: 1
  RGB_SUPERVISION: False
FUTURE_HORIZON: 6
GPUS: 8
IMAGE:
  AUGMENTATION:
    BLUR_PROB: 0.3
    BLUR_STD: [0.1, 1.7]
    BLUR_WINDOW: 5
    COLOR_JITTER_BRIGHTNESS: 0.3
    COLOR_JITTER_CONTRAST: 0.3
    COLOR_JITTER_HUE: 0.1
    COLOR_JITTER_SATURATION: 0.3
    COLOR_PROB: 0.3
    SHARPEN_FACTOR: [1, 5]
    SHARPEN_PROB: 0.3
  CAMERA_POSITION: [-1.5, 0.0, 2.0]
  CAMERA_ROTATION: [0.0, 0.0, 0.0]
  CROP: [64, 138, 896, 458]
  FOV: 100
  IMAGENET_MEAN: (0.485, 0.456, 0.406)
  IMAGENET_STD: (0.229, 0.224, 0.225)
  SIZE: (600, 960)
INSTANCE_SEG:
  CENTER_LABEL_SIGMA_PX: 4
  CENTER_LOSS_WEIGHT: 200.0
  IGNORE_INDEX: 255
  OFFSET_LOSS_WEIGHT: 0.1
LIMIT_VAL_BATCHES: 1
LOGGING_INTERVAL: 500
LOG_DIR: /home/wayve/anthony/experiments/rebuttal
LOSSES:
  KL_BALANCING_ALPHA: 0.75
  WEIGHT_ACTION: 1.0
  WEIGHT_INSTANCE: 0.1
  WEIGHT_PROBABILISTIC: 0.001
  WEIGHT_REWARD: 0.1
  WEIGHT_SEGMENTATION: 0.1
MODEL:
  ACTION_DIM: 2
  BEV:
    BACKBONE: resnet18
    CHANNELS: 64
  EMBEDDING_DIM: 512
  ENCODER:
    NAME: resnet18
    OUT_CHANNELS: 64
  MEASUREMENTS:
    COMMAND_CHANNELS: 8
    ENABLED: False
    GPS_CHANNELS: 16
  POLICY:
    
  REWARD:
    ENABLED: False
  ROUTE:
    BACKBONE: resnet18
    CHANNELS: 16
    ENABLED: True
  SPEED:
    CHANNELS: 16
  TRANSITION:
    ACTION_LATENT_DIM: 64
    DROPOUT_PROBABILITY: 0.15
    ENABLED: True
    HIDDEN_STATE_DIM: 1024
    STATE_DIM: 512
    USE_DROPOUT: True
N_WORKERS: 4
OPTIMIZER:
  ACCUMULATE_GRAD_BATCHES: 1
  LR: 0.0001
  WEIGHT_DECAY: 0.01
PRECISION: 16
PRETRAINED:
  PATH: /home/carla/mile.ckpt
RECEPTIVE_FIELD: 6
ROUTE:
  AUGMENTATION_DEGREES: 8.0
  AUGMENTATION_DROPOUT: 0.025
  AUGMENTATION_END_OF_ROUTE: 0.025
  AUGMENTATION_LARGE_ROTATION: 0.025
  AUGMENTATION_SCALE: (0.95, 1.05)
  AUGMENTATION_SHEAR: (0.1, 0.1)
  AUGMENTATION_SMALL_ROTATION: 0.025
  AUGMENTATION_TRANSLATE: (0.1, 0.1)
  SIZE: 64
SAMPLER:
  COMMAND_WEIGHTS: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  ENABLED: False
  N_BINS: 5
  WITH_ACCELERATION: False
  WITH_ROUTE_COMMAND: False
  WITH_STEERING: False
SCHEDULER:
  NAME: OneCycleLR
  PCT_START: 0.2
SEMANTIC_SEG:
  ENABLED: True
  N_CHANNELS: 8
  TOP_K_RATIO: 0.25
  USE_TOP_K: True
  USE_WEIGHTS: True
SPEED:
  NOISE_STD: 1.4
  NORMALISATION: 5.0
STEPS: 50000
TAG: mile_routemap
VAL_CHECK_INTERVAL: 10000
"""

main()
