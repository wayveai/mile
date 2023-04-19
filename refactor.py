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

from stable_baselines3.common.utils import set_random_seed
from utils.profiling_utils import profile
from vector_input_obs_manager import VectorizedInputManager, MyTaskVehicle, \
    TrafficLightHandlerInstance, init_tl_instance

logger = logging.getLogger(__name__)


NUM_AGENTS = 4
FPS = 10


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
    for ev_id in actor_config:
        bp_filter = actor_config[ev_id]['model']
        blueprint = np.random.choice(world.get_blueprint_library().filter(bp_filter))
        blueprint.set_attribute('role_name', ev_id)

        spawn_transform = np.random.choice([x[1] for x in spawn_transforms])

        wp = world_map.get_waypoint(spawn_transform.location)
        spawn_transform.location.z = wp.transform.location.z + 1.321

        carla_vehicle = world.try_spawn_actor(blueprint, spawn_transform)
        world.tick()

        target_transforms = []
        ego_vehicles[ev_id] = MyTaskVehicle(carla_vehicle, target_transforms, spawn_transforms)

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
    #         'speed': {'module': 'actor_state.speed'}, 'gnss': {'module': 'navigation.gnss'},
    #                      'location': [-1.5, 0.0, 2.0], 'rotation': [0.0, 0.0, 0.0]},
    #         'route_plan': {'module': 'navigation.waypoint_plan', 'steps': 20},
    # 'birdview': {
    #     'module': 'birdview.chauffeurnet', 'width_in_pixels': 192*2,
    #     'pixels_ev_to_bottom': 32, 'pixels_per_meter': 5.0,
    #     'history_idx': [-16, -11, -6, -1], 'scale_bbox': True, 'scale_mask_col': 1.0},
    'vectorized': {
        'module': 'vectorized', 'width_in_pixels': 192 * 2,
        'pixels_ev_to_bottom': 32, 'pixels_per_meter': 5.0,
        'history_idx': [-16, -11, -6, -1], 'scale_bbox': True, 'scale_mask_col': 1.0}
}


obs_configs = {
    'hero%d' % i : single_obs_configs for i in range(NUM_AGENTS)
}

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
                bev_rgb = raw_input['vectorized']['rendered']

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

        # self._tm.set_global_distance_to_leading_vehicle(5.0)
        # logger.debug("trafficmanager set_global_distance_to_leading_vehicle")

        set_random_seed(seed, using_cuda=True)
        self._tm.set_random_device_seed(seed)

        self._world.tick()
        self._traffic_light_handler = init_tl_instance(TrafficLightHandlerInstance(), self._world)
        self._zw_handler = ZombieWalkerHandler(self._client)

        self._obs_managers = {}
        for ev_id, ev_obs_configs in obs_configs.items():
            self._obs_managers[ev_id] = {}
            for obs_id, obs_config in ev_obs_configs.items():
                self._obs_managers[ev_id][obs_id] = VectorizedInputManager(obs_config)

        self._timestamp = None

        self.ego_vehicles = {}
        self._world = client.get_world()

    def reset(self):
        num_zombie_walkers = 120
        actor_config =  {'hero%d' %i : {'model': 'vehicle.lincoln.mkz_2017'} for i in range(NUM_AGENTS)}
        self.ego_vehicles, ev_spawn_locations = reset_ego_vehicles(actor_config, self._world)
        self._zw_handler.reset(num_zombie_walkers, ev_spawn_locations)

        for ev_id, ev_actor in self.ego_vehicles.items():
            for obs_id, om in self._obs_managers[ev_id].items():
                om.attach_ego_vehicle(ev_actor)

        self._world.tick()

        obs_dict = self.get_observation()
        return obs_dict

    def get_observation(self):
        obs_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            obs_dict[ev_id] = {}
            for obs_id, om in om_dict.items():
                obs_dict[ev_id][obs_id] = om.get_observation(self._traffic_light_handler)
        return obs_dict

    def step(self, control_dict):
        for ev_id, control in control_dict.items():
            self.ego_vehicles[ev_id].vehicle.apply_control(control)

        self._world.tick()
        return self.get_observation()

main()
