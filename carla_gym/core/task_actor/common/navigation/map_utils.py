"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np
from enum import Enum


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    # VOID = 0
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def get_sampled_topology(map_topology, resolution):
    """
    Accessor for topology.
    This function retrieves topology from the server as a list of
    road segments as pairs of waypoint objects, and processes the
    topology into a list of dictionary objects.

        :return topology: list of dictionary objects with the following attributes
            entry   -   waypoint of entry point of road segment
            entryxyz-   (x,y,z) of entry point of road segment
            exit    -   waypoint of exit point of road segment
            exitxyz -   (x,y,z) of exit point of road segment
            path    -   list of waypoints separated by 1m from entry
                        to exit
    """
    topology = []
    # Retrieving waypoints to construct a detailed topology
    for segment in map_topology:
        wp1, wp2 = segment[0], segment[1]
        l1, l2 = wp1.transform.location, wp2.transform.location
        # Rounding off to avoid floating point imprecision
        x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
        wp1.transform.location, wp2.transform.location = l1, l2
        seg_dict = dict()
        seg_dict['entry'], seg_dict['exit'] = wp1, wp2
        seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
        seg_dict['path'] = []
        endloc = wp2.transform.location
        if wp1.transform.location.distance(endloc) > resolution:
            w = wp1.next(resolution)[0]
            while w.transform.location.distance(endloc) > resolution:
                seg_dict['path'].append(w)
                w = w.next(resolution)[0]
        else:
            seg_dict['path'].append(wp1.next(resolution)[0])
        topology.append(seg_dict)
    return topology
