"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np


class ZombieWalker(object):
    def __init__(self, walker_id, controller_id, world):

        self._walker = world.get_actor(walker_id)
        self._controller = world.get_actor(controller_id)

        self._controller.start()
        self._controller.go_to_location(world.get_random_location_from_navigation())
        self._controller.set_max_speed(1 + np.random.random())


    def clean(self):
        self._controller.stop()
        self._controller.destroy()
        self._walker.destroy()
