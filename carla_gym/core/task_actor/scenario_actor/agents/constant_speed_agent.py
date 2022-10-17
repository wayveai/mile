"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np
from .utils.local_planner import LocalPlanner


class ConstantSpeedAgent(object):
    def __init__(self, scenario_vehicle, hero_vehicles, target_speed=0.0, max_skip=20, success_dist=5.0):
        self._scenario_vehicle = scenario_vehicle
        self._dest_transform = scenario_vehicle.dest_transform
        self._success_dist = success_dist

        self._local_planner = LocalPlanner(target_speed=target_speed)

    def get_action(self):
        transform = self._scenario_vehicle.vehicle.get_transform()

        if transform.location.distance(self._dest_transform.location) < self._success_dist:
            throttle, steer, brake = 0.0, 0.0, 1.0
        else:
            route_plan = self._scenario_vehicle.route_plan
            velocity = self._scenario_vehicle.vehicle.get_velocity()
            # ego_vehicle_speed
            forward_vec = transform.get_forward_vector()
            vel = np.array([velocity.x, velocity.y, velocity.z])
            f_vec = np.array([forward_vec.x, forward_vec.y, forward_vec.z])
            forward_speed = np.dot(vel, f_vec)
            speed = np.linalg.norm(vel)
            throttle, steer, brake = self._local_planner.run_step(route_plan, transform, forward_speed)

        return np.array([throttle, steer, brake], dtype=np.float64)
