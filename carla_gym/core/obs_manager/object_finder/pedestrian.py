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
        "module": "object_finder.pedestrian",
        "distance_threshold": 50.0,
        "max_detection_number": 5
    }
    """

    def __init__(self, obs_configs):
        self._max_detection_number = obs_configs['max_detection_number']
        self._distance_threshold = obs_configs['distance_threshold']
        self._parent_actor = None
        self._world = None

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
                 low=0, high=5, shape=(self._max_detection_number, 3),
                 dtype=np.float32),
             'absolute_velocity': spaces.Box(
                 low=-5, high=5, shape=(self._max_detection_number, 3),
                 dtype=np.float32),
             'on_sidewalk': spaces.MultiBinary(self._max_detection_number),
             'road_id': spaces.Box(
                low=0, high=5000, shape=(self._max_detection_number, 1),
                dtype=np.int8),
             'lane_id': spaces.Box(
                low=-20, high=20, shape=(self._max_detection_number, 1),
                dtype=np.int8)})

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor
        self._world = parent_actor.vehicle.get_world()
        self._map = self._world.get_map()

    def get_observation(self):
        ev_transform = self._parent_actor.vehicle.get_transform()
        ev_location = ev_transform.location
        def dist_to_actor(w): return w.get_location().distance(ev_location)

        surrounding_pedestrians = []
        pedestrian_list = self._world.get_actors().filter("*walker.pedestrian*")
        for pedestrian in pedestrian_list:
            if dist_to_actor(pedestrian) <= self._distance_threshold:
                surrounding_pedestrians.append(pedestrian)

        sorted_surrounding_pedestrians = sorted(surrounding_pedestrians, key=dist_to_actor)

        location, rotation, absolute_velocity = trans_utils.get_loc_rot_vel_in_ev(
            sorted_surrounding_pedestrians, ev_transform)

        binary_mask, extent, on_sidewalk, road_id, lane_id = [], [], [], [], []
        for ped in sorted_surrounding_pedestrians[:self._max_detection_number]:
            binary_mask.append(1)

            bbox_extent = ped.bounding_box.extent
            extent.append([bbox_extent.x, bbox_extent.y, bbox_extent.z])

            loc = ped.get_location()
            wp = self._map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
            if wp is None:
                on_sidewalk.append(1)
            else:
                on_sidewalk.append(0)
            wp = self._map.get_waypoint(loc)
            road_id.append(wp.road_id)
            lane_id.append(wp.lane_id)

        for i in range(self._max_detection_number - len(binary_mask)):
            binary_mask.append(0)
            location.append([0, 0, 0])
            rotation.append([0, 0, 0])
            absolute_velocity.append([0, 0, 0])
            extent.append([0, 0, 0])
            on_sidewalk.append(0)
            road_id.append(0)
            lane_id.append(0)

        obs_dict = {
            'frame': self._world.get_snapshot().frame,
            'binary_mask': np.array(binary_mask, dtype=np.int8),
            'location': np.array(location, dtype=np.float32),
            'rotation': np.array(rotation, dtype=np.float32),
            'absolute_velocity': np.array(absolute_velocity, dtype=np.float32),
            'extent': np.array(extent, dtype=np.float32),
            'on_sidewalk': np.array(on_sidewalk, dtype=np.int8),
            'road_id': np.array(road_id, dtype=np.int8),
            'lane_id': np.array(lane_id, dtype=np.int8)
        }

        return obs_dict

    def clean(self):
        self._parent_actor = None
        self._world = None
        self._map = None

    # self._debug_draw(sorted_surrounding_pedestrians)
    def _debug_draw(self, pedestrian_list):
        # self._world.debug.draw_point(
        #     ev_location + carla.Location(z=2.0),
        #     color=carla.Color(g=255),
        #     life_time=0.1)
        # extent = carla.Vector3D(x=5.0, y=5.0, z=0.0)
        # box = carla.BoundingBox(extent=extent, location=ev_location+ carla.Location(z=1.0))
        # box = self._parent_actor.vehicle.bounding_box
        # box.location += ev_location
        # self._world.debug.draw_box(box, rotation=self._parent_actor.vehicle.get_transform(
        # ).rotation, color=carla.Color(g=255), life_time=0.05)
        for ped in pedestrian_list:
            self._world.debug.draw_point(
                ped.get_location(),
                color=carla.Color(b=255),
                life_time=0.1)
