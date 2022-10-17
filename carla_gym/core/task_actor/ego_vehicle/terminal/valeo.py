"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np
from collections import deque
import carla

from carla_gym.core.obs_manager.object_finder.vehicle import ObsManager as OmVehicle
from carla_gym.core.obs_manager.object_finder.pedestrian import ObsManager as OmPedestrian
from carla_gym.utils.hazard_actor import lbc_hazard_vehicle, lbc_hazard_walker
from carla_gym.utils.traffic_light import TrafficLightHandler


class Valeo(object):
    '''
    Follow valeo paper as close as possible
    '''

    def __init__(self, ego_vehicle, exploration_suggest=True, eval_mode=False):
        self._ego_vehicle = ego_vehicle
        self._exploration_suggest = exploration_suggest

        self.om_vehicle = OmVehicle({'max_detection_number': 10, 'distance_threshold': 15})
        self.om_pedestrian = OmPedestrian({'max_detection_number': 10, 'distance_threshold': 15})
        self.om_vehicle.attach_ego_vehicle(self._ego_vehicle)
        self.om_pedestrian.attach_ego_vehicle(self._ego_vehicle)

        self._vehicle_stuck_step = 100
        self._vehicle_stuck_counter = 0
        self._speed_queue = deque(maxlen=10)
        self._tl_offset = -0.8 * self._ego_vehicle.vehicle.bounding_box.extent.x
        self._last_lat_dist = 0.0
        self._min_thresh_lat_dist = 3.5

        self._eval_mode = eval_mode
        self._eval_time = 1200

    def get(self, timestamp):
        # Done condition 1: vehicle stuck
        ev_vel = self._ego_vehicle.vehicle.get_velocity()
        ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))
        self._speed_queue.append(ev_speed)
        obs_vehicle = self.om_vehicle.get_observation()
        obs_pedestrian = self.om_pedestrian.get_observation()
        hazard_vehicle_loc = lbc_hazard_vehicle(obs_vehicle, proximity_threshold=9.5)
        hazard_ped_loc = lbc_hazard_walker(obs_pedestrian, proximity_threshold=9.5)

        light_state, light_loc, _ = TrafficLightHandler.get_light_state(self._ego_vehicle.vehicle,
                                                                        offset=self._tl_offset, dist_threshold=18.0)

        is_free_road = (hazard_vehicle_loc is None) and (hazard_ped_loc is None) \
            and (light_state is None or light_state == carla.TrafficLightState.Green)

        if is_free_road and np.mean(self._speed_queue) < 1.0:
            self._vehicle_stuck_counter += 1
        if np.mean(self._speed_queue) >= 1.0:
            self._vehicle_stuck_counter = 0

        c_vehicle_stuck = self._vehicle_stuck_counter >= self._vehicle_stuck_step

        # Done condition 2: lateral distance too large
        ev_loc = self._ego_vehicle.vehicle.get_location()
        wp_transform = self._ego_vehicle.get_route_transform()
        d_vec = ev_loc - wp_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = wp_transform.rotation.get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)
        lat_dist = np.abs(np.dot(np_wp_unit_right, np_d_vec))

        if lat_dist - self._last_lat_dist > 0.8:
            thresh_lat_dist = lat_dist + 0.5
        else:
            thresh_lat_dist = max(self._min_thresh_lat_dist, self._last_lat_dist)
        c_lat_dist = lat_dist > thresh_lat_dist + 1e-2
        self._last_lat_dist = lat_dist

        # Done condition 3: running red light
        c_run_rl = self._ego_vehicle.info_criteria['run_red_light'] is not None
        # Done condition 4: collision
        c_collision = self._ego_vehicle.info_criteria['collision'] is not None
        # Done condition 5: run stop sign
        if self._ego_vehicle.info_criteria['run_stop_sign'] is not None \
                and self._ego_vehicle.info_criteria['run_stop_sign']['event'] == 'run':
            c_run_stop = True
        else:
            c_run_stop = False

        # Done condition 6: vehicle blocked
        c_blocked = self._ego_vehicle.info_criteria['blocked'] is not None

        # endless env: timeout means succeed
        if self._eval_mode:
            timeout = timestamp['relative_simulation_time'] > self._eval_time
        else:
            timeout = False

        done = c_vehicle_stuck or c_lat_dist or c_run_rl or c_collision or c_run_stop or c_blocked or timeout

        # terminal reward
        terminal_reward = 0.0
        if done:
            terminal_reward = -1.0
        if c_run_rl or c_collision or c_run_stop:
            terminal_reward -= ev_speed

        # terminal guide
        exploration_suggest = {
            'n_steps': 0,
            'suggest': ('', '')
        }
        if self._exploration_suggest:
            if c_vehicle_stuck or c_blocked:
                exploration_suggest['n_steps'] = 100
                exploration_suggest['suggest'] = ('go', '')
            if c_lat_dist:
                exploration_suggest['n_steps'] = 100
                exploration_suggest['suggest'] = ('', 'turn')
            if c_run_rl or c_collision or c_run_stop:
                exploration_suggest['n_steps'] = 100
                exploration_suggest['suggest'] = ('stop', '')

        # debug info
        if hazard_vehicle_loc is None:
            txt_hazard_veh = '[]'
        else:
            txt_hazard_veh = np.array2string(hazard_vehicle_loc[0:2], precision=1, separator=',', suppress_small=True)
        if hazard_ped_loc is None:
            txt_hazard_ped = '[]'
        else:
            txt_hazard_ped = np.array2string(hazard_ped_loc[0:2], precision=1, separator=',', suppress_small=True)
        if light_loc is None:
            txt_hazard_rl = '[]'
        else:
            txt_hazard_rl = np.array2string(light_loc[0:2], precision=1, separator=',', suppress_small=True)

        debug_texts = [
            f'{self._vehicle_stuck_counter:3}/{self._vehicle_stuck_step}'
            f' fre:{int(is_free_road)} stu:{int(c_vehicle_stuck)} blo:{int(c_blocked)}',
            f'v:{txt_hazard_veh} p:{txt_hazard_ped} {light_state}{txt_hazard_rl}',
            f'ev: {int(self._eval_mode)} col:{int(c_collision)} red:{int(c_run_rl)} st:{int(c_run_stop)}',
            f'latd:{int(c_lat_dist)}, {lat_dist:.2f}/{thresh_lat_dist:.2f}',
            f"[{exploration_suggest['n_steps']} {exploration_suggest['suggest']}]"
        ]
        terminal_debug = {
            'exploration_suggest': exploration_suggest,
            'debug_texts': debug_texts
        }
        return done, timeout, terminal_reward, terminal_debug
