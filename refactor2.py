import subprocess
import os
import time


from trivial_input_obs_manager import reconstruct_bev
from utils import display_utils

import logging
import numpy as np
import carla
import random

from carla_gym.core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler

from utils.profiling_utils import profile


logger = logging.getLogger(__name__)


NUM_AGENTS = 10
FPS = 10
PEDESTRIANS = 120
SIMULATE_CAR_PHYSICS = True
SIMULATE_PED_PHYSICS = False
CAR_THROTTLE=0.1
ITERATIONS=100


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


def reset_ego_vehicles(actor_config, world, simulate_physics):
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
        carla_vehicle.set_simulate_physics(simulate_physics)
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



class CarlaMultiAgentEnv:
    def __init__(self, num_agents, carla_map, host, port, seed):
        client = carla.Client(host, port)
        client.set_timeout(10.0)

        self._client = client
        self._world = client.load_world(carla_map)

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS
        settings.deterministic_ragdolls = True
        self._world.apply_settings(settings)

        settings = self._world.get_settings()
        settings.no_rendering_mode = True
        self._world.apply_settings(settings)

        random.seed(seed)
        np.random.seed(seed)

        self._world.tick()
        self._zw_handler = ZombieWalkerHandler(self._client)
        self._world = client.get_world()

        actor_config =  [{'model': 'vehicle.lincoln.mkz_2017'} for _ in range(num_agents)]
        self._agent_id_shift = len(self._world.get_level_bbs(carla.CityObjectLabel.Car))
        self.ego_vehicles, ev_spawn_locations = reset_ego_vehicles(actor_config, self._world, SIMULATE_CAR_PHYSICS)
        self._zw_handler.reset(PEDESTRIANS, ev_spawn_locations)
        for w in self._zw_handler.zombie_walkers.values():
            w._walker.set_simulate_physics(SIMULATE_PED_PHYSICS)

        self._world.tick()

    def get_observation(self):
        return dict(
            vehicle_bbox_list=self._world.get_level_bbs(carla.CityObjectLabel.Car),
            walker_bbox_list=[]
            # walker_bbox_list=self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians)
        )

    def tick_world(self, commands):
        self._client.apply_batch_sync(commands, True)

    def close(self):
        self._zw_handler.clean()


def main(num_agents):
    _ = CarlaServerManager('/home/carla/CarlaUE4.sh', port=2000, fps=10, display=False, t_sleep=10)
    print(f'Creating environment with {num_agents} ego vehicles')
    carla_map = 'Town01'
    env = CarlaMultiAgentEnv(num_agents, carla_map=carla_map, host='localhost', port=2000, seed=2021)
    timestamps = []
    full_history = []
    control = carla.VehicleControl(throttle=CAR_THROTTLE, steer=0, brake=0.)
    commands = [carla.command.ApplyVehicleControl(v.id, control) for v in env.ego_vehicles]
    print('starting the loop')
    with profile(enable=False):
        for counter in range(ITERATIONS):
            env.tick_world(commands)
            obs = env.get_observation()

            full_history.append(obs)
            timestamps.append(time.time())
            if counter % 50 == 0:
                print(f"Finished {counter} interations")

    dt = np.median(np.diff(timestamps))
    print(f"dt={dt:.2f}, FPS={1. / dt:.1f}")

    # for agent_id in range(num_agents):
    #     print('Rendering for ', agent_id)
    #     agent_index = agent_id + env._agent_id_shift
    #     images = reconstruct_bev(full_history, agent_index, carla_map)
    #     display_utils.make_video_in_temp(images)


main(NUM_AGENTS)
main(NUM_AGENTS*2)
