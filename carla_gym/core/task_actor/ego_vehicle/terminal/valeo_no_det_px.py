"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np


class ValeoNoDetPx(object):
    '''
    Follow valeo paper as close as possible
    '''

    def __init__(self, ego_vehicle, exploration_suggest=True, eval_mode=False):
        self._ego_vehicle = ego_vehicle
        self._exploration_suggest = exploration_suggest

        self._last_lat_dist = 0.0
        self._min_thresh_lat_dist = 3.5

        self._eval_mode = eval_mode
        self._eval_time = 1200

    def get(self, timestamp):
        # Done condition 1: vehicle blocked
        c_blocked = self._ego_vehicle.info_criteria['blocked'] is not None

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

        # Done condition 6: collision_px
        if self._eval_mode:
            c_collision_px = False
        else:
            c_collision_px = self._ego_vehicle.collision_px

        # endless env: timeout means succeed
        if self._eval_mode:
            timeout = timestamp['relative_simulation_time'] > self._eval_time
        else:
            timeout = False

        done = c_blocked or c_lat_dist or c_run_rl or c_collision or c_run_stop or c_collision_px or timeout

        # terminal reward
        terminal_reward = 0.0
        if done:
            terminal_reward = -1.0
        if c_run_rl or c_collision or c_run_stop or c_collision_px:
            ev_vel = self._ego_vehicle.vehicle.get_velocity()
            ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))
            terminal_reward -= ev_speed

        # terminal guide
        exploration_suggest = {
            'n_steps': 0,
            'suggest': ('', '')
        }
        if self._exploration_suggest:
            if c_blocked:
                exploration_suggest['n_steps'] = 100
                exploration_suggest['suggest'] = ('go', '')
            if c_lat_dist:
                exploration_suggest['n_steps'] = 100
                exploration_suggest['suggest'] = ('go', 'turn')
            if c_run_rl or c_collision or c_run_stop or c_collision_px:
                exploration_suggest['n_steps'] = 100
                exploration_suggest['suggest'] = ('stop', '')

        # debug info

        debug_texts = [
            f'ev: {int(self._eval_mode)} blo:{int(c_blocked)} to:{int(timeout)}',
            f'c_px:{int(c_collision_px)} col:{int(c_collision)} red:{int(c_run_rl)} st:{int(c_run_stop)}',
            f"latd:{int(c_lat_dist)}, {lat_dist:.2f}/{thresh_lat_dist:.2f}, "
            f"[{exploration_suggest['n_steps']} {exploration_suggest['suggest']}]"
        ]
        terminal_debug = {
            'exploration_suggest': exploration_suggest,
            'debug_texts': debug_texts
        }
        return done, timeout, terminal_reward, terminal_debug
