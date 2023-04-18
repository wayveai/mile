import subprocess
import os
import time
import PIL
import PIL.Image
from utils import display_utils

import logging
import numpy as np
import carla

from carla_gym.core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler
from carla_gym.core.zombie_vehicle.zombie_vehicle_handler import ZombieVehicleHandler
from carla_gym.core.obs_manager.obs_manager_handler import ObsManagerHandler
from carla_gym.core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from carla_gym.core.task_actor.scenario_actor.scenario_actor_handler import ScenarioActorHandler
from carla_gym.utils.traffic_light import TrafficLightHandler
from stable_baselines3.common.utils import set_random_seed
from mile.constants import CARLA_FPS
from utils.profiling_utils import profile

logger = logging.getLogger(__name__)


NUM_AGENTS = 2


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



def build_all_tasks(num_zombie_vehicles, num_zombie_walkers):
    weather = 'ClearNoon'

    actor_configs_dict = {
        'ego_vehicles': {
            'hero%d' %i : {'model': 'vehicle.lincoln.mkz_2017'} for i in range(NUM_AGENTS)
        }
    }
    route_descriptions_dict = {
        'ego_vehicles': {
            'hero%d' %i : [] for i in range(NUM_AGENTS)
        }
    }
    endless_dict = {
        'ego_vehicles': {
            'hero%d' %i : True for i in range(NUM_AGENTS)
        }
    }
    all_tasks = []
    task = {
        'weather': weather,
        'description_folder': 'None',
        'route_id': 0,
        'num_zombie_vehicles': num_zombie_vehicles,
        'num_zombie_walkers': num_zombie_walkers,
        'ego_vehicles': {
            'routes': route_descriptions_dict['ego_vehicles'],
            'actors': actor_configs_dict['ego_vehicles'],
            'endless': endless_dict['ego_vehicles']
        },
        'scenario_actors': {},
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



single_obs_configs = {
        #         'speed': {'module': 'actor_state.speed'}, 'gnss': {'module': 'navigation.gnss'},
        #                      'location': [-1.5, 0.0, 2.0], 'rotation': [0.0, 0.0, 0.0]},
        #         'route_plan': {'module': 'navigation.waypoint_plan', 'steps': 20},
        'birdview': {
            'module': 'birdview.chauffeurnet', 'width_in_pixels': 192*2,
            'pixels_ev_to_bottom': 32, 'pixels_per_meter': 5.0,
            'history_idx': [-16, -11, -6, -1], 'scale_bbox': True, 'scale_mask_col': 1.0}
}


obs_configs = {
    'hero%d' % i : single_obs_configs for i in range(NUM_AGENTS)
}

reward_configs = {'hero%d' % i: {'entry_point': 'reward.valeo_action:ValeoAction'} for i in range(NUM_AGENTS)}
terminal_configs = {'hero%d' % i: {'entry_point': 'terminal.leaderboard:Leaderboard'} for i in range(NUM_AGENTS)}
env_configs = {'carla_map': 'Town01', 'routes_group': None, 'weather_group': 'new'}


def main():
    # tasks = build_all_tasks(env_configs['carla_map'])
    tasks = build_all_tasks(0, 100)

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
    actor_id = 'hero0'

    obs = env.reset(0)
    debug_frames = [[] for i in range(NUM_AGENTS)]
    timestamps = []
    print('starting the loop')
    with profile(enable=False):
        for counter in range(200):
            if counter % 50 == 0:
                print(counter)
                print(obs.keys())

            for agent_i in range(NUM_AGENTS):

                raw_input = obs[f"hero{agent_i}"]
                #         front_rgb = raw_input['central_rgb']['data']
                bev_rgb = raw_input['birdview']['rendered']

                #         img = overlay_images(downsample(front_rgb, 0.6), bev_rgb, (20, 20))
                if bev_rgb is not None:
                    debug_frames[agent_i].append(bev_rgb)

            control_dict = {
                'hero%d' % i: carla.VehicleControl(throttle=0.8, steer=0, brake=0.) for i in range(NUM_AGENTS)
            }
            # get observations
            obs = env.step(control_dict)

            timestamps.append(time.time())

    dt = np.median(np.diff(timestamps))
    print(f"dt={dt:.2f}, FPS={1. / dt:.1f}")
    print(len(debug_frames[0]))
    for frames in debug_frames:
        if len(frames):
            display_utils.make_video_in_temp(frames)

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
        self._observation_handler = ObsManagerHandler(obs_configs)
        # this contains all info related to reward, traffic lights violations etc
        self._ego_vehicle_handler = EgoVehicleHandler(self._client, reward_configs, terminal_configs)
        self._zw_handler = ZombieWalkerHandler(self._client)
        self._zv_handler = ZombieVehicleHandler(self._client, tm_port=self._tm.get_port())
        self._scenario_actor_handler = ScenarioActorHandler(self._client)

    def reset(self, task_idx):
        task = self._all_tasks[task_idx].copy()

        ev_spawn_locations = self._ego_vehicle_handler.reset(task['ego_vehicles'])
        self._scenario_actor_handler.reset(task['scenario_actors'], self._ego_vehicle_handler.ego_vehicles)
        self._zw_handler.reset(task['num_zombie_walkers'], ev_spawn_locations)
        self._zv_handler.reset(task['num_zombie_vehicles'], ev_spawn_locations)
        self._observation_handler.reset(self._ego_vehicle_handler.ego_vehicles)

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
        _, _, _ = self._ego_vehicle_handler.tick(timestamp)
        obs_dict = self._observation_handler.get_observation(timestamp)
        return obs_dict

    def step(self, control_dict):
        self._ego_vehicle_handler.apply_control(control_dict)
        self._scenario_actor_handler.tick()
        # tick world
        self._world.tick()

        # update timestamp
        snap_shot = self._world.get_snapshot()
        self._timestamp['step'] = snap_shot.timestamp.frame-self._timestamp['start_frame']
        self._timestamp['frame'] = snap_shot.timestamp.frame
        self._timestamp['wall_time'] = snap_shot.timestamp.platform_timestamp
        self._timestamp['relative_wall_time'] = self._timestamp['wall_time'] - self._timestamp['start_wall_time']
        self._timestamp['simulation_time'] = snap_shot.timestamp.elapsed_seconds
        self._timestamp['relative_simulation_time'] = self._timestamp['simulation_time'] \
            - self._timestamp['start_simulation_time']

        return self._observation_handler.get_observation(self.timestamp)

    @property
    def timestamp(self):
        return self._timestamp.copy()

main()
