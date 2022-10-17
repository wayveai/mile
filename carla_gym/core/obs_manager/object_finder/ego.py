"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np
import carla
from gym import spaces
from carla_gym.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):
    def __init__(self, obs_configs):
        self._parent_actor = None
        self._map = None
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'actor_location': spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32),
            'actor_rotation': spaces.Box(low=-180, high=180, shape=(3,), dtype=np.float32),
            'waypoint_location': spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32),
            'waypoint_rotation': spaces.Box(low=-180, high=180, shape=(3,), dtype=np.float32),
            'road_id': spaces.Discrete(5000),
            'section_id': spaces.Discrete(5000),
            'lane_id': spaces.Box(low=-20, high=20, shape=(1,), dtype=np.int8),
            'is_junction':  spaces.Discrete(2),
            'lane_change': spaces.Discrete(4),
            'extent': spaces.Box(low=0, high=20, shape=(3,), dtype=np.float32),
            'speed_limit': spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32)
        })

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor
        self._map = parent_actor.vehicle.get_world().get_map()

    def get_observation(self):

        actor_transform = self._parent_actor.vehicle.get_transform()

        actor_location = [actor_transform.location.x,
                          actor_transform.location.y,
                          actor_transform.location.z]
        actor_rotation = [actor_transform.rotation.roll,
                          actor_transform.rotation.pitch,
                          actor_transform.rotation.yaw]

        actor_wp = self._map.get_waypoint(actor_transform.location, project_to_road=True,
                                          lane_type=carla.LaneType.Driving)

        waypoint_location = [actor_wp.transform.location.x,
                             actor_wp.transform.location.y,
                             actor_wp.transform.location.z]
        waypoint_rotation = [actor_wp.transform.rotation.roll,
                             actor_wp.transform.rotation.pitch,
                             actor_wp.transform.rotation.yaw]

        extent = self._parent_actor.vehicle.bounding_box.extent
        speed_limit = self._parent_actor.vehicle.get_speed_limit()

        obs_dict = {
            'actor_location': np.array(actor_location, dtype=np.float32),
            'actor_rotation': np.array(actor_rotation, dtype=np.float32),
            'waypoint_location': np.array(waypoint_location, dtype=np.float32),
            'waypoint_rotation': np.array(waypoint_rotation, dtype=np.float32),
            'road_id': int(actor_wp.road_id),
            'section_id': int(actor_wp.section_id),
            'lane_id': int(actor_wp.lane_id),
            'is_junction':  int(actor_wp.is_junction),
            'lane_change': int(actor_wp.lane_change),
            'extent': np.array([extent.x, extent.y, extent.z], dtype=np.float32),
            'speed_limit': np.float32(speed_limit)
        }

        return obs_dict

    def clean(self):
        self._parent_actor = None
        self._map = None
