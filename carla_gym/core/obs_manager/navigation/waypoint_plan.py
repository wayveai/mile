"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np
from gym import spaces

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase

import carla_gym.utils.transforms as trans_utils


class ObsManager(ObsManagerBase):
    """
    Template config
    "obs_configs" = {
        "module": "navigation.waypoint_plan",
        "steps": 10
    }
    [command, loc_x, loc_y]
    """

    def __init__(self, obs_configs):
        self._steps = obs_configs['steps']
        self._parent_actor = None
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'location': spaces.Box(low=-100, high=1000, shape=(self._steps, 2), dtype=np.float32),
            'command': spaces.Box(low=-1, high=6, shape=(self._steps,), dtype=np.uint8),
            'road_id': spaces.Box(low=0, high=6000, shape=(self._steps,), dtype=np.uint8),
            'lane_id': spaces.Box(low=-20, high=20, shape=(self._steps,), dtype=np.int8),
            'is_junction': spaces.MultiBinary(self._steps)})

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor
        self._world = self._parent_actor.vehicle.get_world()

    def get_observation(self):
        ev_transform = self._parent_actor.vehicle.get_transform()

        route_plan = self._parent_actor.route_plan

        route_length = len(route_plan)
        location_list = []
        command_list = []
        road_id = []
        lane_id = []
        is_junction = []
        for i in range(self._steps):
            if i < route_length:
                waypoint, road_option = route_plan[i]
            else:
                waypoint, road_option = route_plan[-1]

            wp_location_world_coord = waypoint.transform.location
            wp_location_actor_coord = trans_utils.loc_global_to_ref(wp_location_world_coord, ev_transform)
            location_list.append([wp_location_actor_coord.x, wp_location_actor_coord.y])
            command_list.append(road_option.value)
            road_id.append(waypoint.road_id)
            lane_id.append(waypoint.lane_id)
            is_junction.append(waypoint.is_junction)

        obs_dict = {
            'location': np.array(location_list, dtype=np.float32),
            'command': np.array(command_list, dtype=np.int8),
            'road_id': np.array(road_id, dtype=np.int8),
            'lane_id': np.array(lane_id, dtype=np.int8),
            'is_junction': np.array(is_junction, dtype=np.int8)
        }
        return obs_dict

    def clean(self):
        self._parent_actor = None
        self._world = None

# VOID = 0
# LEFT = 1
# RIGHT = 2
# STRAIGHT = 3
# LANEFOLLOW = 4
# CHANGELANELEFT = 5
# CHANGELANERIGHT = 6
