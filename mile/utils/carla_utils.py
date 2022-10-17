import numpy as np


def get_vector3(vec3):
    return np.array([vec3.x, vec3.y, vec3.z])


def get_wheel_base(vehicle):
    physics_control = vehicle.get_physics_control()
    wheel_base = np.linalg.norm(
        get_vector3(physics_control.wheels[0].position - physics_control.wheels[2].position)) / 100
    return wheel_base


def convert_steer_to_curvature(steer, wheel_base):
    return -np.tan(steer) / wheel_base
