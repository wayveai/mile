"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np


def is_within_distance_ahead(target_location, max_distance, up_angle_th=60):
    distance = np.linalg.norm(target_location[0:2])
    if distance < 0.001:
        return True
    if distance > max_distance:
        return False
    x = target_location[0]
    y = target_location[1]
    angle = np.rad2deg(np.arctan2(y, x))
    return abs(angle) < up_angle_th


def lbc_hazard_vehicle(obs_surrounding_vehicles, ev_speed=None, proximity_threshold=9.5):
    for i, is_valid in enumerate(obs_surrounding_vehicles['binary_mask']):
        if not is_valid:
            continue

        sv_yaw = obs_surrounding_vehicles['rotation'][i][2]
        same_heading = abs(sv_yaw) <= 150

        sv_loc = obs_surrounding_vehicles['location'][i]
        with_distance_ahead = is_within_distance_ahead(sv_loc, proximity_threshold, up_angle_th=45)
        if same_heading and with_distance_ahead:
            return sv_loc
    return None


def lbc_hazard_walker(obs_surrounding_pedestrians, ev_speed=None, proximity_threshold=9.5):
    for i, is_valid in enumerate(obs_surrounding_pedestrians['binary_mask']):
        if not is_valid:
            continue
        if int(obs_surrounding_pedestrians['on_sidewalk'][i]) == 1:
            continue

        ped_loc = obs_surrounding_pedestrians['location'][i]

        dist = np.linalg.norm(ped_loc)
        degree = 162 / (np.clip(dist, 1.5, 10.5)+0.3)

        if is_within_distance_ahead(ped_loc, proximity_threshold, up_angle_th=degree):
            return ped_loc
    return None


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


def challenge_hazard_walker(obs_surrounding_pedestrians, ev_speed=None):
    p1 = np.float32([0, 0])
    v1 = np.float32([10, 0])

    for i, is_valid in enumerate(obs_surrounding_pedestrians['binary_mask']):
        if not is_valid:
            continue

        ped_loc = obs_surrounding_pedestrians['location'][i]
        ped_yaw = obs_surrounding_pedestrians['rotation'][i][2]
        ped_vel = obs_surrounding_pedestrians['absolute_velocity'][i]

        v2_hat = np.float32([np.cos(np.radians(ped_yaw)), np.sin(np.radians(ped_yaw))])
        s2 = np.linalg.norm(ped_vel)

        if s2 < 0.05:
            v2_hat *= s2

        p2 = -3.0 * v2_hat + ped_loc[0:2]
        v2 = 8.0 * v2_hat

        collides, collision_point = get_collision(p1, v1, p2, v2)

        if collides:
            return ped_loc
    return None


def challenge_hazard_vehicle(obs_surrounding_vehicles, ev_speed):
    # np.linalg.norm(_numpy(self._vehicle.get_velocity())
    o1 = np.float32([1, 0])
    p1 = np.float32([0, 0])
    s1 = max(9.5, 2.0 * ev_speed)
    v1_hat = o1
    v1 = s1 * v1_hat

    for i, is_valid in enumerate(obs_surrounding_vehicles['binary_mask']):
        if not is_valid:
            continue

        sv_loc = obs_surrounding_vehicles['location'][i]
        sv_yaw = obs_surrounding_vehicles['rotation'][i][2]
        sv_vel = obs_surrounding_vehicles['absolute_velocity'][i]

        o2 = np.float32([np.cos(np.radians(sv_yaw)), np.sin(np.radians(sv_yaw))])
        p2 = sv_loc[0:2]
        s2 = max(5.0, 2.0 * np.linalg.norm(sv_vel[0:2]))
        v2_hat = o2
        v2 = s2 * v2_hat

        p2_p1 = p2 - p1
        distance = np.linalg.norm(p2_p1)
        p2_p1_hat = p2_p1 / (distance + 1e-4)

        angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
        angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

        if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1:
            continue

        return sv_loc

    return None


def behavior_hazard_vehicle(ego_vehicle, actors, route_plan, proximity_th, up_angle_th, lane_offset=0, at_junction=False):
    '''
    ego_vehicle: input_data['ego_vehicle']
    actors: input_data['surrounding_vehicles']
    route_plan: input_data['route_plan']
    '''
    # Get the right offset
    if ego_vehicle['lane_id'] < 0 and lane_offset != 0:
        lane_offset *= -1

    for i, is_valid in enumerate(actors['binary_mask']):
        if not is_valid:
            continue

        if not at_junction and (actors['road_id'][i] != ego_vehicle['road_id'] or
                                actors['lane_id'][i] != ego_vehicle['lane_id'] + lane_offset):

            next_road_id = route_plan['road_id'][5]
            next_lane_id = route_plan['lane_id'][5]

            if actors['road_id'][i] != next_road_id or actors['lane_id'][i] != next_lane_id + lane_offset:
                continue

        if is_within_distance_ahead(actors['location'][i], proximity_th, up_angle_th):
            return i
    return None


def behavior_hazard_walker(ego_vehicle, actors, route_plan, proximity_th, up_angle_th, lane_offset=0, at_junction=False):
    '''
    ego_vehicle: input_data['ego_vehicle']
    actors: input_data['surrounding_vehicles']
    route_plan: input_data['route_plan']
    '''
    # Get the right offset
    if ego_vehicle['lane_id'] < 0 and lane_offset != 0:
        lane_offset *= -1

    for i, is_valid in enumerate(actors['binary_mask']):
        if not is_valid:
            continue

        if int(actors['on_sidewalk'][i]) == 1:
            continue

        if not at_junction and (actors['road_id'][i] != ego_vehicle['road_id'] or
                                actors['lane_id'][i] != ego_vehicle['lane_id'] + lane_offset):

            next_road_id = route_plan['road_id'][5]
            next_lane_id = route_plan['lane_id'][5]

            if actors['road_id'][i] != next_road_id or actors['lane_id'][i] != next_lane_id + lane_offset:
                continue

        if is_within_distance_ahead(actors['location'][i], proximity_th, up_angle_th):
            return i
    return None
