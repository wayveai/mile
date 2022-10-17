"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np
import weakref
import carla
from gym import spaces
from queue import Queue, Empty

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from carla_gym.core.task_actor.common.navigation.map_utils import RoadOption

from mile.data.dataset_utils import preprocess_gps


class ObsManager(ObsManagerBase):

    def __init__(self, obs_configs):

        self._gnss_sensor = None
        self._imu_sensor = None
        self._queue_timeout = 10.0
        self._gnss_queue = None
        self._imu_queue = None

        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        # accelerometer: m/s2
        # gyroscope: rad/s2
        # compass: rad wrt. north
        imu_low = np.array([-1e6, -1e6, -1e6, -1e6, -1e6, -1e6, 0], dtype=np.float32)
        imu_high = np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 2*np.pi], dtype=np.float32)

        self.obs_space = spaces.Dict({
            'gnss': spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            'imu': spaces.Box(low=imu_low, high=imu_high, dtype=np.float32),
            'target_gps': spaces.Box(low=-1e3, high=1e3, shape=(3,), dtype=np.float32),
            'command': spaces.Box(low=-1, high=6, shape=(1,), dtype=np.int8),
            'target_gps_next': spaces.Box(low=-1e3, high=1e3, shape=(3,), dtype=np.float32),
            'command_next': spaces.Box(low=-1, high=6, shape=(1,), dtype=np.int8),
        })

    def attach_ego_vehicle(self, parent_actor):
        self._world = parent_actor.vehicle.get_world()
        self._parent_actor = parent_actor
        self._idx = -1
        self._gnss_queue = Queue()
        self._imu_queue = Queue()

        # gnss sensor
        bp = self._world.get_blueprint_library().find('sensor.other.gnss')
        bp.set_attribute('noise_alt_stddev', str(0.000005))
        bp.set_attribute('noise_lat_stddev', str(0.000005))
        bp.set_attribute('noise_lon_stddev', str(0.000005))
        bp.set_attribute('noise_alt_bias', str(0.0))
        bp.set_attribute('noise_lat_bias', str(0.0))
        bp.set_attribute('noise_lon_bias', str(0.0))
        sensor_location = carla.Location()
        sensor_rotation = carla.Rotation()
        sensor_transform = carla.Transform(location=sensor_location, rotation=sensor_rotation)
        self._gnss_sensor = self._world.spawn_actor(bp, sensor_transform, attach_to=parent_actor.vehicle)
        weak_self = weakref.ref(self)
        self._gnss_sensor.listen(lambda gnss_data: self._parse_gnss(weak_self, gnss_data))

        # imu sensor
        bp = self._world.get_blueprint_library().find('sensor.other.imu')
        bp.set_attribute('noise_accel_stddev_x', str(0.001))
        bp.set_attribute('noise_accel_stddev_y', str(0.001))
        bp.set_attribute('noise_accel_stddev_z', str(0.015))
        bp.set_attribute('noise_gyro_stddev_x', str(0.001))
        bp.set_attribute('noise_gyro_stddev_y', str(0.001))
        bp.set_attribute('noise_gyro_stddev_z', str(0.001))
        sensor_location = carla.Location()
        sensor_rotation = carla.Rotation()
        sensor_transform = carla.Transform(location=sensor_location, rotation=sensor_rotation)
        self._imu_sensor = self._world.spawn_actor(bp, sensor_transform, attach_to=parent_actor.vehicle)
        weak_self = weakref.ref(self)
        self._imu_sensor.listen(lambda imu_data: self._parse_imu(weak_self, imu_data))

    def get_observation(self):
        snap_shot = self._world.get_snapshot()
        assert self._gnss_queue.qsize() <= 1
        assert self._imu_queue.qsize() <= 1

        # get gnss
        try: 
            frame, gnss_data = self._gnss_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception('gnss sensor took too long!')

        # get imu
        try: 
            frame, imu_data = self._imu_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception('imu sensor took too long!')

        # target gps
        global_plan_gps = self._parent_actor.global_plan_gps

        next_gps, _ = global_plan_gps[self._idx+1]

        loc_in_ev = preprocess_gps(gnss_data, next_gps, imu_data)
        if np.sqrt(loc_in_ev.x**2+loc_in_ev.y**2) < 12.0 and loc_in_ev.x < 0.0:
            self._idx += 1

        self._idx = min(self._idx, len(global_plan_gps)-2)

        _, road_option_0 = global_plan_gps[max(0, self._idx)]
        gps_point, road_option_1 = global_plan_gps[self._idx+1]
        # Gps waypoint after the immediate next waypoint.
        gps_point2, road_option_2 = global_plan_gps[min(len(global_plan_gps) - 1, self._idx + 2)]

        if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option = road_option_1
        else:
            road_option = road_option_0

        # Handle road option for next next waypoint
        if (road_option_1 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_2 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option_next = road_option_2
        else:
            road_option_next = road_option_1

        obs = {'gnss': gnss_data,
               'imu': imu_data,
               'target_gps': np.array(gps_point, dtype=np.float32),
               'command': np.array([road_option.value], dtype=np.int8),
               'target_gps_next': np.array(gps_point2, dtype=np.float32),
               'command_next': np.array([road_option_next.value], dtype=np.int8),
               }
        return obs

    def clean(self):
        if self._imu_sensor and self._imu_sensor.is_alive:
            self._imu_sensor.stop()
            self._imu_sensor.destroy()
        self._imu_sensor = None

        if self._gnss_sensor and self._gnss_sensor.is_alive:
            self._gnss_sensor.stop()
            self._gnss_sensor.destroy()
        self._gnss_sensor = None

        self._world = None
        self._parent_actor = None

        self._gnss_queue = None
        self._imu_queue = None

    @staticmethod
    def _parse_gnss(weak_self, gnss_data):
        self = weak_self()
        data = np.array([gnss_data.latitude,
                         gnss_data.longitude,
                         gnss_data.altitude], dtype=np.float32)
        self._gnss_queue.put((gnss_data.frame, data))

    @staticmethod
    def _parse_imu(weak_self, imu_data):
        self = weak_self()
        data = np.array([imu_data.accelerometer.x,
                         imu_data.accelerometer.y,
                         imu_data.accelerometer.z,
                         imu_data.gyroscope.x,
                         imu_data.gyroscope.y,
                         imu_data.gyroscope.z,
                         imu_data.compass,
                         ], dtype=np.float32)
        self._imu_queue.put((imu_data.frame, data))
