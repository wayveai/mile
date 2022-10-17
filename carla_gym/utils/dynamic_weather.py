# modified from https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/dynamic_weather.py

import carla
import numpy as np
from mile.constants import CARLA_FPS

WEATHERS = [
    carla.WeatherParameters.Default,

    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.ClearSunset,

    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.CloudySunset,

    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetSunset,

    carla.WeatherParameters.MidRainyNoon,
    carla.WeatherParameters.MidRainSunset,

    carla.WeatherParameters.WetCloudyNoon,
    carla.WeatherParameters.WetCloudySunset,

    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.HardRainSunset,

    carla.WeatherParameters.SoftRainNoon,
    carla.WeatherParameters.SoftRainSunset,
]


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = np.random.uniform(0.0, 2.0*np.pi)

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * np.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (55 * np.sin(self._t)) + 35

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class WeatherHandler(object):
    def __init__(self, world):
        self._world = world
        self._dynamic = False

    def reset(self, cfg_weather):
        if hasattr(carla.WeatherParameters, cfg_weather):
            self._world.set_weather(getattr(carla.WeatherParameters, cfg_weather))
            self._dynamic = False
        elif 'dynamic' in cfg_weather:
            self._weather = np.random.choice(WEATHERS)
            self._sun = Sun(self._weather.sun_azimuth_angle, self._weather.sun_altitude_angle)
            self._storm = Storm(self._weather.precipitation)
            self._dynamic = True
            l = cfg_weather.split('_')
            if len(l) == 2:
                self._speed_factor = float(l[1])
            else:
                self._speed_factor = 1.0
            self.tick(1.0 / CARLA_FPS)
        else:
            self._world.set_weather('ClearNoon')
            self._dynamic = False

    def tick(self, delta_seconds):
        if self._dynamic:
            self._sun.tick(delta_seconds * self._speed_factor)
            self._storm.tick(delta_seconds * self._speed_factor)
            self._weather.cloudiness = self._storm.clouds
            self._weather.precipitation = self._storm.rain
            self._weather.precipitation_deposits = self._storm.puddles
            self._weather.wind_intensity = self._storm.wind
            self._weather.fog_density = self._storm.fog
            self._weather.wetness = self._storm.wetness
            self._weather.sun_azimuth_angle = self._sun.azimuth
            self._weather.sun_altitude_angle = self._sun.altitude
            self._world.set_weather(self._weather)

    def clean(self):
        if self._dynamic:
            self._weather = None
            self._sun = None
            self._storm = None
        self._dynamic = False
