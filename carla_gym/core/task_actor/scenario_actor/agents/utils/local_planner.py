"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

from enum import Enum
import numpy as np

from .controller import PIDController
import carla_gym.utils.transforms as trans_utils


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations
    when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):

    def __init__(self, target_speed=0.0,
                 longitudinal_pid_params=[0.5, 0.025, 0.1],
                 lateral_pid_params=[0.75, 0.05, 0.0],
                 threshold_before=7.5,
                 threshold_after=5.0):

        self._target_speed = target_speed
        self._speed_pid = PIDController(longitudinal_pid_params)
        self._turn_pid = PIDController(lateral_pid_params)
        self._threshold_before = threshold_before
        self._threshold_after = threshold_after
        self._max_skip = 20

        self._last_command = 4

    def run_step(self, route_plan, actor_transform, actor_speed):
        target_index = -1
        for i, (waypoint, road_option) in enumerate(route_plan[0:self._max_skip]):
            if self._last_command == 4 and road_option.value != 4:
                threshold = self._threshold_before
            else:
                threshold = self._threshold_after

            distance = waypoint.transform.location.distance(actor_transform.location)
            if distance < threshold:
                self._last_command = road_option.value
                target_index = i

        if target_index < len(route_plan)-1:
            target_index += 1
        target_command = route_plan[target_index][1]
        target_location_world_coord = route_plan[target_index][0].transform.location
        target_location_actor_coord = trans_utils.loc_global_to_ref(target_location_world_coord, actor_transform)

        # steer
        x = target_location_actor_coord.x
        y = target_location_actor_coord.y
        theta = np.arctan2(y, x)
        steer = self._turn_pid.step(theta)

        # throttle
        target_speed = self._target_speed
        if target_command not in [3, 4]:
            target_speed *= 0.75
        delta = target_speed - actor_speed
        throttle = self._speed_pid.step(delta)

        # brake
        brake = 0.0

        # clip
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)

        return throttle, steer, brake
