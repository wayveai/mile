import subprocess
import os
import time


from trivial_input_obs_manager import reconstruct_bev
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


def set_sync_mode(world, sync):
    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / FPS
    settings.deterministic_ragdolls = True
    world.apply_settings(settings)



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
    ego_vehicles = []
    for ev_id in range(len(actor_config)):
        bp_filter = actor_config[ev_id]['model']
        blueprint = np.random.choice(world.get_blueprint_library().filter(bp_filter))
        blueprint.set_attribute('role_name', str(ev_id))

        carla_vehicle = None
        for attempt in range(100):
            spawn_transform = np.random.choice([x[1] for x in spawn_transforms])

            wp = world_map.get_waypoint(spawn_transform.location)
            spawn_transform.location.z = wp.transform.location.z + 1.321

            carla_vehicle = world.try_spawn_actor(blueprint, spawn_transform)
            if carla_vehicle is not None:
                break

            print(f"Failed to spawn {ev_id} (attempt={attempt})")

        assert carla_vehicle is not None
        world.tick()

        ego_vehicles.append(carla_vehicle)

        ev_spawn_locations.append(carla_vehicle.get_location())

    return ego_vehicles, ev_spawn_locations


class CarlaServerManager:
    def __init__(self, carla_sh_str, port=2000, fps=25, display=False, t_sleep=5):
        kill_process = subprocess.Popen(f'fuser -k {port}/tcp', shell=True)
        kill_process.wait()
        print(f"Killed Carla Servers on port {port}!")

        cmd = f'bash {carla_sh_str} ' \
              f'-fps={fps} -nosound -quality-level=Low -carla-rpc-port={port}'
        if not display:
            cmd += ' -RenderOffScreen'

        self._server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        time.sleep(t_sleep)


def main():
    _ = CarlaServerManager(
        '/home/carla/CarlaUE4.sh', port=2000, fps=10, display=False, t_sleep=10
    )

    print('Creating environment')
    carla_map = 'Town01'
    env = CarlaMultiAgentEnv(
        carla_map=carla_map,
        host='localhost',
        port=2000,
        seed=2021,
        no_rendering=True,
    )
    timestamps = []
    throttle = 0.8
    vehicle_ids = [v.id for v in env.ego_vehicles]
    control_dict = [carla.VehicleControl(throttle=throttle, steer=0, brake=0.) for i in range(NUM_AGENTS)]
    print('starting the loop')
    full_history = []
    commands = []
    for actor_id, control in zip(vehicle_ids, control_dict):
        commands.append(carla.command.ApplyVehicleControl(actor_id, control))

    with profile(enable=True):
        for counter in range(100):
            if counter % 50 == 0:
                print(counter)

            env.tick_world(commands)
            obs = env.get_observation()

            full_history.append(obs)
            timestamps.append(time.time())

    dt = np.median(np.diff(timestamps))
    print(f"dt={dt:.2f}, FPS={1. / dt:.1f}")

    for agent_id in range(NUM_AGENTS):
        print('Rendering for ', agent_id)
        agent_index = agent_id + env._agent_id_shift
        images = reconstruct_bev(full_history, agent_index, carla_map)
        display_utils.make_video_in_temp(images)


class CarlaMultiAgentEnv:
    def __init__(self, carla_map, host, port, seed, no_rendering):
        client = carla.Client(host, port)
        client.set_timeout(10.0)

        self._client = client
        self._world = client.load_world(carla_map)


        set_sync_mode(self._world, True)
        set_no_rendering_mode(self._world, no_rendering)
        set_random_seed(seed, using_cuda=True)

        self._world.tick()
        self._zw_handler = ZombieWalkerHandler(self._client)
        self._world = client.get_world()

        actor_config =  [{'model': 'vehicle.lincoln.mkz_2017'} for i in range(NUM_AGENTS)]
        self._agent_id_shift = len(self._world.get_level_bbs(carla.CityObjectLabel.Car))
        self.ego_vehicles, ev_spawn_locations = reset_ego_vehicles(actor_config, self._world)
        self._zw_handler.reset(PEDESTRIANS, ev_spawn_locations)
        self._world.tick()

    def get_observation(self):
        return dict(
            vehicle_bbox_list=self._world.get_level_bbs(carla.CityObjectLabel.Car),
            walker_bbox_list=self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians)
        )

    def tick_world(self, commands):

        results = self._client.apply_batch_sync(commands, False)
        self._world.tick()


main()
