"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

# base class for observation managers


class ObsManagerBase(object):

    def __init__(self):
        self._define_obs_space()

    def _define_obs_space(self):
        raise NotImplementedError

    def attach_ego_vehicle(self, parent_actor):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError
