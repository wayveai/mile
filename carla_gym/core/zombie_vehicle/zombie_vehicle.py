"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np
import carla

class ZombieVehicle(object):
    def __init__(self, actor_id, world):
        self._vehicle = world.get_actor(actor_id)

    def teleport_to(self, transform):
        self._vehicle.set_transform(transform)
        self._vehicle.set_velocity(carla.Vector3D())

    def clean(self):
        # self._vehicle.set_autopilot(False)
        self._vehicle.destroy()
