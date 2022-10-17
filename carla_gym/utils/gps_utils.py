"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import math

EARTH_RADIUS_EQUA = 6378137.0


def gps2xyz(lat, lon, z, lat_ref=49.0, lon_ref=8.0):
    # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)

    mx = lon / 180.0 * (math.pi * EARTH_RADIUS_EQUA * scale)
    my = math.log(math.tan((lat+90.0)*math.pi/360.0))*(EARTH_RADIUS_EQUA * scale)

    x = mx - scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - my

    return x, y, z


def xyz2gps(x, y, z, lat_ref=49.0, lon_ref=8.0):
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += x
    my -= y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    return lat, lon, z
