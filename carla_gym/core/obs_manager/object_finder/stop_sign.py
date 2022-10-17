"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

from gym import spaces
import carla
from carla_gym.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):

    def __init__(self, obs_configs):
        self._parent_actor = None
        self._distance_threshold = obs_configs['distance_threshold']

        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'at_stop_sign': spaces.Discrete(2)
        })

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

    def get_observation(self):
        ev_loc = self._parent_actor.vehicle.get_location()
        stop_sign = self._parent_actor.criteria_stop._target_stop_sign

        at_stop_sign = 0
        if (stop_sign is not None) and (not self._parent_actor.criteria_stop._stop_completed):
            stop_t = stop_sign.get_transform()
            stop_loc = stop_t.transform(stop_sign.trigger_volume.location)

            if carla.Location(stop_loc).distance(ev_loc) < self._distance_threshold:
                at_stop_sign = 1

        obs = {'at_stop_sign': at_stop_sign}
        return obs

    def clean(self):
        self._parent_actor = None
