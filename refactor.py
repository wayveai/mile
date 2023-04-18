import subprocess
import os
import time
import PIL
import PIL.Image
from utils import display_utils

import logging
import gym
import numpy as np
import carla


from carla_gym import CARLA_GYM_ROOT_DIR
from carla_gym.utils import config_utils
import json
from carla_gym.core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler
from carla_gym.core.zombie_vehicle.zombie_vehicle_handler import ZombieVehicleHandler
from carla_gym.core.obs_manager.obs_manager_handler import ObsManagerHandler
from carla_gym.core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from carla_gym.core.task_actor.scenario_actor.scenario_actor_handler import ScenarioActorHandler
from carla_gym.utils.traffic_light import TrafficLightHandler
from carla_gym.utils.dynamic_weather import WeatherHandler
from stable_baselines3.common.utils import set_random_seed
from mile.constants import CARLA_FPS

logger = logging.getLogger(__name__)


def set_no_rendering_mode(world, no_rendering):
    settings = world.get_settings()
    settings.no_rendering_mode = no_rendering
    world.apply_settings(settings)


def set_sync_mode(world, tm, sync):
    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / CARLA_FPS
    settings.deterministic_ragdolls = True
    world.apply_settings(settings)
    tm.set_synchronous_mode(sync)



def build_all_tasks(carla_map):
    assert carla_map in ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06']
    num_zombie_vehicles = {
        'Town01': 120,
        'Town02': 70,
        'Town03': 70,
        'Town04': 150,
        'Town05': 120,
        'Town06': 120
    }[carla_map]
    num_zombie_walkers = {
        'Town01': 120,
        'Town02': 70,
        'Town03': 70,
        'Town04': 80,
        'Town05': 120,
        'Town06': 80
    }[carla_map]

    num_zombie_vehicles = 0
    num_zombie_walkers = 0

    weather = 'ClearNoon'
    description_folder = CARLA_GYM_ROOT_DIR / 'envs/scenario_descriptions/LeaderBoard' / carla_map

    actor_configs_dict = json.load(open(description_folder / 'actors.json'))
    route_descriptions_dict = config_utils.parse_routes_file(description_folder / 'routes.xml')

    all_tasks = []
    for route_id, route_description in route_descriptions_dict.items():
        task = {
            'weather': weather,
            'description_folder': description_folder,
            'route_id': route_id,
            'num_zombie_vehicles': num_zombie_vehicles,
            'num_zombie_walkers': num_zombie_walkers,
            'ego_vehicles': {
                'routes': route_description['ego_vehicles'],
                'actors': actor_configs_dict['ego_vehicles'],
            },
            'scenario_actors': {
                'routes': route_description['scenario_actors'],
                'actors': actor_configs_dict['scenario_actors']
            } if 'scenario_actors' in actor_configs_dict else {}
        }
        all_tasks.append(task)

    return all_tasks


def kill_carla(port=2005):
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


obs_configs = {
    'hero': {
        #         'speed': {'module': 'actor_state.speed'}, 'gnss': {'module': 'navigation.gnss'},
        #                      'location': [-1.5, 0.0, 2.0], 'rotation': [0.0, 0.0, 0.0]},
        #         'route_plan': {'module': 'navigation.waypoint_plan', 'steps': 20},
        'birdview': {
            'module': 'birdview.chauffeurnet', 'width_in_pixels': 192,
            'pixels_ev_to_bottom': 32, 'pixels_per_meter': 5.0,
            'history_idx': [-16, -11, -6, -1], 'scale_bbox': True, 'scale_mask_col': 1.0}},
    #     'hero': {}
}

reward_configs = {'hero': {'entry_point': 'reward.valeo_action:ValeoAction'}}
terminal_configs = {'hero': {'entry_point': 'terminal.leaderboard:Leaderboard'}}
env_configs = {'carla_map': 'Town01', 'routes_group': None, 'weather_group': 'new'}


def main():
    tasks = build_all_tasks(env_configs['carla_map'])

    server_manager = CarlaServerManager(
        '/home/carla/CarlaUE4.sh', port=2000, fps=10, display=False, t_sleep=10
    )
    server_manager.start()

    print('Crating environment')
    env = CarlaMultiAgentEnv(
        carla_map=env_configs['carla_map'],
        host='localhost',
        port=2000,
        seed=2021,
        no_rendering=True,
        obs_configs=obs_configs,
        reward_configs=reward_configs,
        terminal_configs=terminal_configs,
        all_tasks=tasks
    )
    actor_id = 'hero'

    obs = env.reset(0)
    debug_frames = []
    timestamps = []
    print('starting the loop')
    for counter in range(200):
        raw_input = obs[actor_id]
        if counter % 50 == 0:
            print(counter)
            settings = env._world.get_settings()
            print(settings.no_rendering_mode)
            print(raw_input.keys())

        #         front_rgb = raw_input['central_rgb']['data']
        bev_rgb = raw_input['birdview']['rendered']

        #         img = overlay_images(downsample(front_rgb, 0.6), bev_rgb, (20, 20))
        if bev_rgb is not None:
            debug_frames.append(bev_rgb)

        control_dict = {
            actor_id: carla.VehicleControl(throttle=0.8, steer=0, brake=0.)
        }

        env._ev_handler.apply_control(control_dict)
        env._sa_handler.tick()
        # tick world
        env._world.tick()

        env._ev_handler.tick(env._timestamp.copy())

        # get observations
        obs = env._om_handler.get_observation(env._timestamp.copy())

        timestamps.append(time.time())

    dt = np.median(np.diff(timestamps))
    print(f"dt={dt:.2f}, FPS={1. / dt:.1f}")
    print(len(debug_frames))
    # if len(debug_frames):
    #     display_utils.make_video_in_temp(debug_frames)

    # env.close()
    # server_manager.stop()


class CarlaMultiAgentEnv:
    def __init__(self, carla_map, host, port, seed, no_rendering,
                 obs_configs, reward_configs, terminal_configs, all_tasks):
        self._all_tasks = all_tasks

        client = carla.Client(host, port)
        client.set_timeout(10.0)

        self._client = client
        self._world = client.load_world(carla_map)
        self._tm = client.get_trafficmanager(port+6000)

        set_sync_mode(self._world, self._tm, True)
        set_no_rendering_mode(self._world, no_rendering)

        # self._tm.set_hybrid_physics_mode(True)

        # self._tm.set_global_distance_to_leading_vehicle(5.0)
        # logger.debug("trafficmanager set_global_distance_to_leading_vehicle")

        set_random_seed(seed, using_cuda=True)
        self._tm.set_random_device_seed(seed)

        self._world.tick()

        # register traffic lights
        TrafficLightHandler.reset(self._world)

        # define observation spaces exposed to agent
        self._om_handler = ObsManagerHandler(obs_configs)
        # this contains all info related to reward, traffic lights violations etc
        self._ev_handler = EgoVehicleHandler(self._client, reward_configs, terminal_configs)
        self._zw_handler = ZombieWalkerHandler(self._client)
        self._zv_handler = ZombieVehicleHandler(self._client, tm_port=self._tm.get_port())
        self._sa_handler = ScenarioActorHandler(self._client)

    def reset(self, task_idx):

        task = self._all_tasks[task_idx].copy()

        ev_spawn_locations = self._ev_handler.reset(task['ego_vehicles'])
        self._sa_handler.reset(task['scenario_actors'], self._ev_handler.ego_vehicles)
        self._zw_handler.reset(task['num_zombie_walkers'], ev_spawn_locations)
        self._zv_handler.reset(task['num_zombie_vehicles'], ev_spawn_locations)
        self._om_handler.reset(self._ev_handler.ego_vehicles)

        self._world.tick()

        snap_shot = self._world.get_snapshot()
        self._timestamp = {
            'step': 0,
            'frame': snap_shot.timestamp.frame,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }
        timestamp = self._timestamp.copy()
        _, _, _ = self._ev_handler.tick(timestamp)
        obs_dict = self._om_handler.get_observation(timestamp)
        return obs_dict

main()
