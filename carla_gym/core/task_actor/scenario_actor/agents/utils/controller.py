"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

from collections import deque


class PIDController(object):
    def __init__(self, pid_list, n=30):
        self._K_P, self._K_I, self._K_D = pid_list

        self._dt = 1.0 / 10.0
        self._window = deque(maxlen=n)

    def reset(self):
        self._window.clear()

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = sum(self._window) * self._dt
            derivative = (self._window[-1] - self._window[-2]) / self._dt
        else:
            integral = 0.0
            derivative = 0.0

        control = 0.0
        control += self._K_P * error
        control += self._K_I * integral
        control += self._K_D * derivative

        return control
