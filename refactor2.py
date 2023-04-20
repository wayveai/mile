import subprocess
import os
import time
import PIL
import PIL.Image

from trivial_input_obs_manager import TrivialInputManager
from utils import display_utils

import logging
import numpy as np
import carla

from carla_gym.core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler

from stable_baselines3.common.utils import set_random_seed
from utils.profiling_utils import profile


logger = logging.getLogger(__name__)


NUM_AGENTS = 4
FPS = 10
PEDESTRIANS = 120


def set_no_rendering_mode(world, no_rendering):
    settings = world.get_settings()
    settings.no_rendering_mode = no_rendering
    world.apply_settings(settings)


def set_sync_mode(world, tm, sync):
    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / FPS
    settings.deterministic_ragdolls = True
    world.apply_settings(settings)
    tm.set_synchronous_mode(sync)


def _get_spawn_points(c_map):
    all_spawn_points = c_map.get_spawn_points()

    spawn_transforms = []
    for trans in all_spawn_points:
        wp = c_map.get_waypoint(trans.location)

        if wp.is_junction:
            wp_prev = wp
            while wp_prev.is_junction:
                wp_prev = wp_prev.previous(1.0)[0]
            spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
            if c_map.name == 'Town03' and (wp_prev.road_id == 44):
                for _ in range(100):
                    spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
        else:
            spawn_transforms.append([wp.road_id, wp.transform])
            if c_map.name == 'Town03' and (wp.road_id == 44):
                for _ in range(100):
                    spawn_transforms.append([wp.road_id, wp.transform])

    return spawn_transforms



def reset_ego_vehicles(actor_config, world):
    world_map = world.get_map()
    spawn_transforms = _get_spawn_points(world_map)

    ev_spawn_locations = []
    ego_vehicles = {}
    for ev_id, _ in enumerate(actor_config):
        bp_filter = actor_config[ev_id]['model']
        blueprint = np.random.choice(world.get_blueprint_library().filter(bp_filter))
        blueprint.set_attribute('role_name', str(ev_id))

        spawn_transform = np.random.choice([x[1] for x in spawn_transforms])

        wp = world_map.get_waypoint(spawn_transform.location)
        spawn_transform.location.z = wp.transform.location.z + 1.321

        carla_vehicle = world.try_spawn_actor(blueprint, spawn_transform)
        world.tick()

        ego_vehicles[ev_id] = carla_vehicle

        ev_spawn_locations.append(carla_vehicle.get_location())

    return ego_vehicles, ev_spawn_locations



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
    'width_in_pixels': 192 * 2,
    'pixels_ev_to_bottom': 32, 'pixels_per_meter': 5.0,
    'history_idx': [-16, -11, -6, -1], 'scale_bbox': True, 'scale_mask_col': 1.0
}


obs_configs = [single_obs_configs for i in range(NUM_AGENTS)]

def main():
    server_manager = CarlaServerManager(
        '/home/carla/CarlaUE4.sh', port=2000, fps=10, display=False, t_sleep=10
    )
    server_manager.start()

    print('Creating environment')
    env = CarlaMultiAgentEnv(
        carla_map='Town01',
        host='localhost',
        port=2000,
        seed=2021,
        no_rendering=True,
        obs_configs=obs_configs,
    )

    obs = env.reset()
    timestamps = []
    control_dict = [carla.VehicleControl(throttle=0.8, steer=0, brake=0.) for i in range(NUM_AGENTS)
        # i: carla.VehicleControl(throttle=0.0, steer=0, brake=0.) for i in range(NUM_AGENTS)
    ]
    print('starting the loop')
    with profile(enable=True):
        for counter in range(100):
            if counter % 50 == 0:
                print(counter)
                print(obs.keys())
            # get observations
            obs = env.step(control_dict)

            timestamps.append(time.time())

    dt = np.median(np.diff(timestamps))
    print(f"dt={dt:.2f}, FPS={1. / dt:.1f}")

    reconstructed_bevs = env.reconstruct_bev()
    debug_frames = []
    for i in range(NUM_AGENTS):
        agent_bevs = reconstructed_bevs[i]
        debug_frames.append([el['rendered'] for el in agent_bevs])

    for frames in debug_frames:
        if len(frames):
            display_utils.make_video_in_temp(frames)


class CarlaMultiAgentEnv:
    def __init__(self, carla_map, host, port, seed, no_rendering,
                 obs_configs):

        client = carla.Client(host, port)
        client.set_timeout(10.0)

        self._client = client
        self._world = client.load_world(carla_map)
        self._tm = client.get_trafficmanager(port+6000)

        set_sync_mode(self._world, self._tm, True)
        set_no_rendering_mode(self._world, no_rendering)

        # self._tm.set_hybrid_physics_mode(True)

        set_random_seed(seed, using_cuda=True)
        self._tm.set_random_device_seed(seed)

        self._world.tick()
        self._zw_handler = ZombieWalkerHandler(self._client)

        self._obs_managers = {}
        for ev_id, obs_config in enumerate(obs_configs):
            self._obs_managers[ev_id] = TrivialInputManager(obs_config)

        self._timestamp = None

        self.ego_vehicles = {}
        self._world = client.get_world()

    def reset(self):
        actor_config =  [{'model': 'vehicle.lincoln.mkz_2017'} for i in range(NUM_AGENTS)]
        agent_id_shift = len(self._world.get_level_bbs(carla.CityObjectLabel.Car))
        self.ego_vehicles, ev_spawn_locations = reset_ego_vehicles(actor_config, self._world)
        self._zw_handler.reset(PEDESTRIANS, ev_spawn_locations)

        for ev_id, ev_actor in self.ego_vehicles.items():
            self._obs_managers[ev_id].attach_ego_vehicle(ev_actor, agent_id_shift)

        self._world.tick()

        obs_dict = self.get_observation()
        return obs_dict

    def get_observation(self):
        obs_dict = {}
        for ev_id, om in self._obs_managers.items():
            obs_dict[ev_id] = om.get_observation()
        return obs_dict

    def step(self, control_dict):
        self._apply_control(control_dict)
        self._tick_world()
        return self.get_observation()

    def _apply_control(self, control_dict):
        for ev_id, control in enumerate(control_dict):
            self.ego_vehicles[ev_id].apply_control(control)

    def _tick_world(self):
        self._world.tick()

    def reconstruct_bev(self):
        result = {}
        for ev_id, om in self._obs_managers.items():
            print('Rendering for ', ev_id)
            result[ev_id] = om.reconstruct_bev()
        return result

main()
