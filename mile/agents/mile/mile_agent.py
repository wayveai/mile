"""Adapted from https://github.com/zhejz/carla-roach CC-BY-NC 4.0 license."""

import logging
from collections import deque

import torch
from omegaconf import OmegaConf
from torchmetrics import JaccardIndex

from carla_gym.utils.config_utils import load_entry_point
from mile.constants import CARLA_FPS, DISPLAY_SEGMENTATION
from mile.data.dataset import calculate_geometry_from_config
from mile.data.dataset_utils import preprocess_birdview_and_routemap, preprocess_measurements, calculate_birdview_labels
from mile.trainer import WorldModelTrainer


class MileAgent:
    def __init__(self, path_to_conf_file='config_agent.yaml'):
        self._logger = logging.getLogger(__name__)
        self._render_dict = None
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        cfg = OmegaConf.load(path_to_conf_file)

        # load checkpoint from wandb
        self._ckpt = None

        cfg = OmegaConf.to_container(cfg)
        self._obs_configs = cfg['obs_configs']
        # for debug view
        self._obs_configs['route_plan'] = {'module': 'navigation.waypoint_plan', 'steps': 20}
        wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])

        # prepare policy
        self.input_buffer_size = 1
        if cfg['ckpt'] is not None:
            trainer = WorldModelTrainer.load_from_checkpoint(cfg['ckpt'], pretrained_path=cfg['ckpt'])
            print(f'Loading world model weights from {cfg["ckpt"]}')
            self._policy = trainer.to('cuda')
            game_frequency = CARLA_FPS
            model_stride_sec = self._policy.cfg.DATASET.STRIDE_SEC
            receptive_field = trainer.model.receptive_field
            n_image_per_stride = int(game_frequency * model_stride_sec)

            self.input_buffer_size = (receptive_field - 1) * n_image_per_stride + 1
            self.sequence_indices = range(0, self.input_buffer_size, n_image_per_stride)

        self._env_wrapper = wrapper_class(cfg=self._policy.cfg)

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
        self.cfg = cfg

        # Custom metrics
        if self._policy.cfg.SEMANTIC_SEG.ENABLED and DISPLAY_SEGMENTATION:
            self.iou = JaccardIndex(task='multiclass', num_classes=self._policy.cfg.SEMANTIC_SEG.N_CHANNELS).cuda()
            self.real_time_iou = JaccardIndex(
                task='multiclass', num_classes=self._policy.cfg.SEMANTIC_SEG.N_CHANNELS, compute_on_step=True,
            ).cuda()

        if self.cfg['online_deployment']:
            print('Online deployment')
        else:
            print('Recomputing')

    def run_step(self, input_data, timestamp):
        policy_input = self.preprocess_data(input_data)
        # Forward pass
        with torch.no_grad():
            is_dreaming = False
            if self.cfg['online_deployment']:
                output = self._policy.deployment_forward(policy_input, is_dreaming=is_dreaming)
            else:
                output = self._policy(policy_input, deployment=True)

        actions = torch.cat([output['throttle_brake'], output['steering']], dim=-1)[0, 0].cpu().numpy()
        control = self._env_wrapper.process_act(actions)

        # Populate action queue
        self.action_queue.append(torch.from_numpy(actions).cuda())

        # Metrics
        metrics = self.forward_metrics(policy_input, output)

        self.prepare_rendering(policy_input, output, metrics, timestamp, is_dreaming)

        return control

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

    def prepare_rendering(self, policy_input, output, metrics, timestamp, is_dreaming):
        # For rendering
        self._render_dict = {
            'policy_input': policy_input,
            'obs_configs': self._obs_configs,
            'policy_cfg': self._policy.cfg,
            'metrics': metrics,
        }

        for k, v in output.items():
            self._render_dict[k] = v

        self._render_dict['timestamp'] = timestamp
        self._render_dict['is_dreaming'] = is_dreaming

        self.supervision_dict = {}

    def reset(self, log_file_path):
        # logger
        self._logger.handlers = []
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)

        for v in self.policy_input_queue.values():
            v.clear()

        self.action_queue.clear()

    def render(self, reward_debug, terminal_debug):
        '''
        test render, used in evaluate.py
        '''
        self._render_dict['reward_debug'] = reward_debug
        self._render_dict['terminal_debug'] = terminal_debug
        im_render = self._env_wrapper.im_render(self._render_dict)
        return im_render

    def forward_metrics(self, policy_input, output):
        real_time_metrics = {}
        if self._policy.cfg.SEMANTIC_SEG.ENABLED and DISPLAY_SEGMENTATION:
            with torch.no_grad():
                bev_prediction = output['bev_segmentation_1'].detach()
                bev_prediction = torch.argmax(bev_prediction, dim=2)[:, -1]
                bev_label = policy_input['birdview_label'][:, -1, 0]
                self.iou(bev_prediction.view(-1), bev_label.view(-1))

                real_time_metrics['intersection-over-union'] = self.real_time_iou(bev_prediction, bev_label).mean().item()

        return real_time_metrics

    def compute_metrics(self):
        metrics = {}
        if self._policy.cfg.SEMANTIC_SEG.ENABLED and DISPLAY_SEGMENTATION:
            scores = self.iou.compute()
            metrics['intersection-over-union'] = scores.item()
            self.iou.reset()
        return metrics

    @property
    def obs_configs(self):
        return self._obs_configs
