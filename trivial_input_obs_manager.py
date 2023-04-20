"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import cv2 as cv
import carla
import numpy as np
from pathlib import Path
import h5py


COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_5 = (46, 52, 54)


def tint(color, factor):
    r, g, b = color
    r = int(r + (255-r) * factor)
    g = int(g + (255-g) * factor)
    b = int(b + (255-b) * factor)
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return (r, g, b)


def reconstruct_bev(full_history, agent_index, map_name, obs_config):
    width = int(obs_config['width_in_pixels'])
    pixels_ev_to_bottom = obs_config['pixels_ev_to_bottom']
    pixels_per_meter = obs_config['pixels_per_meter']

    map_dir = Path('/home/carla/mile/carla_gym/core/obs_manager/birdview/maps')
    maps_h5_path = map_dir / (map_name.rsplit('/', 1)[-1] + '.h5')
    with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
        road = np.array(hf['road'], dtype=np.uint8)
        world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)

    results = [
        make_bev_output(
            obs, road, agent_index,
            width, pixels_per_meter, pixels_ev_to_bottom, world_offset
        ) for obs in full_history]
    return results


def make_bev_output(
        item, road, agent_index, width, pixels_per_meter, pixels_ev_to_bottom, world_offset):

    ev_bbox = item['vehicle_bbox_list'][agent_index]
    ev_transform = carla.Transform(ev_bbox.location, ev_bbox.rotation)

    ev_loc = ev_transform.location
    ev_rot = ev_transform.rotation

    M_warp = _get_warp_transform(ev_loc, ev_rot, width, pixels_per_meter, pixels_ev_to_bottom, world_offset)

    vehicles = []
    for bbox in item['vehicle_bbox_list']:
        vehicles.append((carla.Transform(bbox.location, bbox.rotation), carla.Location(), carla.Vector3D(bbox.extent)))

    walkers = []
    for bbox in item['walker_bbox_list']:
        walkers.append(
            (carla.Transform(bbox.location, bbox.rotation), carla.Location(), carla.Vector3D(bbox.extent)))

    vehicle_mask = _get_mask_from_actor_list(vehicles, M_warp, width, pixels_per_meter, world_offset)
    walker_mask = _get_mask_from_actor_list(walkers, M_warp, width, pixels_per_meter, world_offset)

    road_mask = cv.warpAffine(road, M_warp, (width, width)).astype(np.bool)

    ev_mask = _get_mask_from_actor_list(
        [(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp, width, pixels_per_meter, world_offset)

    image = np.zeros([width, width, 3], dtype=np.uint8)
    image[road_mask] = COLOR_ALUMINIUM_5

    image[vehicle_mask] = tint(COLOR_BLUE, 0.2)
    image[walker_mask] = tint(COLOR_CYAN, 0.2)

    image[ev_mask] = COLOR_WHITE
    return image


def _world_to_pixel(location, pixels_per_meter, world_offset):
    """Converts the world coordinates to pixel coordinates"""
    x = pixels_per_meter * (location.x - world_offset[0])
    y = pixels_per_meter * (location.y - world_offset[1])
    return np.array([x, y], dtype=np.float32)


def _get_mask_from_actor_list(actor_list, M_warp, width, pixels_per_meter, world_offset):
    mask = np.zeros([width, width], dtype=np.uint8)
    for actor_transform, bb_loc, bb_ext in actor_list:

        corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                   carla.Location(x=bb_ext.x, y=-bb_ext.y),
                   carla.Location(x=bb_ext.x, y=0),
                   carla.Location(x=bb_ext.x, y=bb_ext.y),
                   carla.Location(x=-bb_ext.x, y=bb_ext.y)]
        corners = [bb_loc + corner for corner in corners]

        corners = [actor_transform.transform(corner) for corner in corners]
        corners_in_pixel = np.array([[_world_to_pixel(corner, pixels_per_meter, world_offset)] for corner in corners])
        corners_warped = cv.transform(corners_in_pixel, M_warp)

        cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
    return mask.astype(np.bool)


def _get_warp_transform(ev_loc, ev_rot, width, pixels_per_meter, pixels_ev_to_bottom, world_offset):
    ev_loc_in_px = _world_to_pixel(ev_loc, pixels_per_meter, world_offset)
    yaw = np.deg2rad(ev_rot.yaw)

    forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
    right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])

    bottom_left = ev_loc_in_px - pixels_ev_to_bottom * forward_vec - (0.5*width) * right_vec
    top_left = ev_loc_in_px + (width-pixels_ev_to_bottom) * forward_vec - (0.5*width) * right_vec
    top_right = ev_loc_in_px + (width-pixels_ev_to_bottom) * forward_vec + (0.5*width) * right_vec

    src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
    dst_pts = np.array([[0, width-1],
                        [0, 0],
                        [width-1, 0]], dtype=np.float32)
    return cv.getAffineTransform(src_pts, dst_pts)



