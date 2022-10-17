"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np
import carla
from gym import spaces
from carla_gym.core.obs_manager.obs_manager import ObsManagerBase

import carla_gym.utils.transforms as trans_utils


class ObsManager(ObsManagerBase):
    """
    Template config
    obs_configs = {
        "module": "object_finder.vehicle",
        "distance_threshold": 50.0,
        "max_detection_number": 5
    }
    """

    def __init__(self, obs_configs):
        self._max_detection_number = obs_configs['max_detection_number']
        self._distance_threshold = obs_configs['distance_threshold']

        self._parent_actor = None
        self._world = None
        self._map = None
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict(
            {'frame': spaces.Discrete(2**32-1),
             'binary_mask': spaces.MultiBinary(self._max_detection_number),
             'location': spaces.Box(
                 low=-self._distance_threshold, high=self._distance_threshold, shape=(self._max_detection_number, 3),
                dtype=np.float32),
             'rotation': spaces.Box(
                low=-180, high=180, shape=(self._max_detection_number, 3),
                dtype=np.float32),
             'extent': spaces.Box(
                low=0, high=20, shape=(self._max_detection_number, 3),
                dtype=np.float32),
             'absolute_velocity': spaces.Box(
                low=-10, high=50, shape=(self._max_detection_number, 3),
                dtype=np.float32),
             'road_id': spaces.Box(
                low=0, high=5000, shape=(self._max_detection_number, 1),
                dtype=np.int8),
             'lane_id': spaces.Box(
                low=-20, high=20, shape=(self._max_detection_number, 1),
                dtype=np.int8)})

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor
        self._world = self._parent_actor.vehicle.get_world()
        self._map = self._world.get_map()

    def get_observation(self):
        ev_transform = self._parent_actor.vehicle.get_transform()
        ev_location = ev_transform.location
        def dist_to_ev(w): return w.get_location().distance(ev_location)

        surrounding_vehicles = []
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        for vehicle in vehicle_list:
            has_different_id = self._parent_actor.vehicle.id != vehicle.id
            is_within_distance = dist_to_ev(vehicle) <= self._distance_threshold
            if has_different_id and is_within_distance:
                surrounding_vehicles.append(vehicle)

        sorted_surrounding_vehicles = sorted(surrounding_vehicles, key=dist_to_ev)

        location, rotation, absolute_velocity = trans_utils.get_loc_rot_vel_in_ev(
            sorted_surrounding_vehicles, ev_transform)

        binary_mask, extent, road_id, lane_id = [], [], [], []
        for sv in sorted_surrounding_vehicles[:self._max_detection_number]:
            binary_mask.append(1)

            bbox_extent = sv.bounding_box.extent
            extent.append([bbox_extent.x, bbox_extent.y, bbox_extent.z])

            loc = sv.get_location()
            wp = self._map.get_waypoint(loc)
            road_id.append(wp.road_id)
            lane_id.append(wp.lane_id)

        for i in range(self._max_detection_number - len(binary_mask)):
            binary_mask.append(0)
            location.append([0, 0, 0])
            rotation.append([0, 0, 0])
            extent.append([0, 0, 0])
            absolute_velocity.append([0, 0, 0])
            road_id.append(0)
            lane_id.append(0)

        obs_dict = {
            'frame': self._world.get_snapshot().frame,
            'binary_mask': np.array(binary_mask, dtype=np.int8),
            'location': np.array(location, dtype=np.float32),
            'rotation': np.array(rotation, dtype=np.float32),
            'extent': np.array(extent, dtype=np.float32),
            'absolute_velocity': np.array(absolute_velocity, dtype=np.float32),
            'road_id': np.array(road_id, dtype=np.int8),
            'lane_id': np.array(lane_id, dtype=np.int8)
        }
        return obs_dict

    def clean(self):
        self._parent_actor = None
        self._world = None
        self._map = None
