"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

from importlib import import_module
from gym import spaces

class ObsManagerHandler(object):

    def __init__(self, obs_configs):
        self._obs_managers = {}
        self._obs_configs = obs_configs
        self._init_obs_managers()

    def get_observation(self, timestamp):
        obs_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            obs_dict[ev_id] = {}
            for obs_id, om in om_dict.items():
                obs_dict[ev_id][obs_id] = om.get_observation()
        return obs_dict

    @property
    def observation_space(self):
        obs_spaces_dict = {}
        for ev_id, om_dict in self._obs_managers.items():
            ev_obs_spaces_dict = {}
            for obs_id, om in om_dict.items():
                ev_obs_spaces_dict[obs_id] = om.obs_space
            obs_spaces_dict[ev_id] = spaces.Dict(ev_obs_spaces_dict)
        return spaces.Dict(obs_spaces_dict)

    def reset(self, ego_vehicles):
        self._init_obs_managers()

        for ev_id, ev_actor in ego_vehicles.items():
            for obs_id, om in self._obs_managers[ev_id].items():
                om.attach_ego_vehicle(ev_actor)

    def clean(self):
        for ev_id, om_dict in self._obs_managers.items():
            for obs_id, om in om_dict.items():
                om.clean()
        self._obs_managers = {}

    def _init_obs_managers(self):
        for ev_id, ev_obs_configs in self._obs_configs.items():
            self._obs_managers[ev_id] = {}
            for obs_id, obs_config in ev_obs_configs.items():
                ObsManager = getattr(import_module('carla_gym.core.obs_manager.'+obs_config["module"]), 'ObsManager')
                self._obs_managers[ev_id][obs_id] = ObsManager(obs_config)
