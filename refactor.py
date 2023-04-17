import cv2
import carla
import gym
import carla_gym
import torch
import numpy as np
import subprocess
import os
import time
import PIL
import PIL.Image
from utils import display_utils

obs_configs = {
    'hero': {
        #         'speed': {'module': 'actor_state.speed'}, 'gnss': {'module': 'navigation.gnss'},
        #         'central_rgb': {'module': 'camera.rgb', 'fov': 100, 'width': 960, 'height': 600,
        #                      'location': [-1.5, 0.0, 2.0], 'rotation': [0.0, 0.0, 0.0]},
        #         'route_plan': {'module': 'navigation.waypoint_plan', 'steps': 20},
        'birdview': {'module': 'birdview.chauffeurnet', 'width_in_pixels': 192,
                     'pixels_ev_to_bottom': 32, 'pixels_per_meter': 5.0,
                     'history_idx': [-16, -11, -6, -1], 'scale_bbox': True, 'scale_mask_col': 1.0}},
    #     'hero': {}
}

reward_configs = {'hero': {'entry_point': 'reward.valeo_action:ValeoAction'}}
terminal_configs = {'hero': {'entry_point': 'terminal.leaderboard:Leaderboard'}}
env_configs = {'carla_map': 'Town02', 'routes_group': None, 'weather_group': 'new'}


def overlay_images(
        background_img: np.ndarray, foreground_img: np.ndarray, position, alpha: float = 1.0
) -> np.ndarray:
    background_img = PIL.Image.fromarray(background_img, 'RGB')
    foreground_img = PIL.Image.fromarray(foreground_img, 'RGB')

    a_channel = int(255 * alpha)
    mask = PIL.Image.new('RGBA', foreground_img.size, (0, 0, 0, a_channel))

    background_img.paste(foreground_img, box=position, mask=mask)
    return np.asarray(background_img, dtype=np.uint8)


def downsample(image, scale_factor):
    pil_image = PIL.Image.fromarray(np.uint8(image))
    downsampled_image = pil_image.resize((int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)),
                                         resample=PIL.Image.LANCZOS)
    return np.array(downsampled_image)


def kill_carla(port=2005):
    # The command below kills ALL carla processes
    # kill_process = subprocess.Popen('killall -9 -r CarlaUE4-Linux', shell=True)

    # This one only kills processes linked to a certain port
    kill_process = subprocess.Popen(f'fuser -k {port}/tcp', shell=True)
    kill_process.wait()
    print(f"Killed Carla Servers on port {port}!")
    time.sleep(1)


class CarlaServerManager:
    def __init__(self, carla_sh_str, port=2000, fps=25, display=False, t_sleep=5):
        self._carla_sh_str = carla_sh_str
        self._port = port
        self._fps = fps
        self._t_sleep = t_sleep
        self._server_process = None
        self._display = display

    def start(self):
        self.stop()
        #         cmd = f'bash {self._carla_sh_str} ' \
        #             f'-fps={self._fps} -nosound -quality-level=Epic -carla-rpc-port={self._port}'
        cmd = f'bash {self._carla_sh_str} ' \
              f'-fps={self._fps} -nosound -quality-level=Low -carla-rpc-port={self._port}'
        if not self._display:
            cmd += ' -RenderOffScreen'

        self._server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        time.sleep(self._t_sleep)

    def stop(self):
        if self._server_process is not None:
            self._server_process.kill()
        kill_carla(self._port)
        time.sleep(self._t_sleep)


def main():
    server_manager = CarlaServerManager(
        '/home/carla/CarlaUE4.sh', port=2000, fps=25, display=False)
    server_manager.start()

    env = gym.make(
        'LeaderBoard-v0',
        obs_configs=obs_configs,
        reward_configs=reward_configs,
        terminal_configs=terminal_configs,
        host='localhost',
        port=2000,
        seed=2021,
        no_rendering=True,
        **env_configs)

    task_idx = 0
    env.set_task_idx(task_idx)
    actor_id = 'hero'

    obs = env.reset()
    debug_frames = []
    timestamps = []
    for counter in range(100):
        raw_input = obs[actor_id]
        if counter % 20 == 0:
            print(counter)
            settings = env._world.get_settings()
            print(settings.no_rendering_mode)
            print(raw_input.keys())

        #         front_rgb = raw_input['central_rgb']['data']
        bev_rgb = raw_input['birdview']['rendered']
        #         img = overlay_images(downsample(front_rgb, 0.6), bev_rgb, (20, 20))
        debug_frames.append(bev_rgb)

        control_dict = {
            actor_id: carla.VehicleControl(throttle=0.8, steer=0, brake=0.)
        }
        obs, reward, done, info = env.step(control_dict)
        timestamps.append(time.time())

    dt = np.median(np.diff(timestamps))
    print(f"dt={dt:.2f}, FPS={1. / dt:.1f}")
    if len(debug_frames):
        display_utils.make_video_in_temp(debug_frames)

    # env.close()
    # server_manager.stop()


main()
