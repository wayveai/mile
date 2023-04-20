"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import cv2 as cv
from pathlib import Path
import h5py
import carla
import numpy as np


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


def _get_surrounding_actors(bbox_list, criterium, scale=None):
    actors = []
    for bbox in bbox_list:
        is_within_distance = criterium(bbox)
        if is_within_distance:
            bb_loc = carla.Location()
            bb_ext = carla.Vector3D(bbox.extent)
            if scale is not None:
                bb_ext = bb_ext * scale
                bb_ext.x = max(bb_ext.x, 0.8)
                bb_ext.y = max(bb_ext.y, 0.8)

            actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
    return actors


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


def image_render(road_mask,
               vehicle_mask, walker_mask, ev_mask, width):
    image = np.zeros([width, width, 3], dtype=np.uint8)
    image[road_mask] = COLOR_ALUMINIUM_5

    image[vehicle_mask] = tint(COLOR_BLUE, 0.2)
    image[walker_mask] = tint(COLOR_CYAN, 0.2)

    image[ev_mask] = COLOR_WHITE
    return image


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


class TrivialInputManager:
    def __init__(self, obs_configs):
        self._width = int(obs_configs['width_in_pixels'])
        self._pixels_ev_to_bottom = obs_configs['pixels_ev_to_bottom']
        self._pixels_per_meter = obs_configs['pixels_per_meter']
        self._history_idx = obs_configs['history_idx']
        self._scale_mask_col = obs_configs.get('scale_mask_col', 1.1)
        self._obs_config = obs_configs

        self._full_history = []

        self._image_channels = 3
        self._vehicle = None
        self._world = None

        self._map_dir = Path(__file__).resolve().parent / 'carla_gym/core/obs_manager/birdview/maps'

    def attach_ego_vehicle(self, parent_actor):
        self._vehicle = parent_actor
        self._world = self._vehicle.get_world()

        maps_h5_path = self._map_dir / (self._world.get_map().name.rsplit('/', 1)[-1] + '.h5')
        with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
            self._road = np.array(hf['road'], dtype=np.uint8)
            self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
            assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))

        self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)

    def get_observation(self):
        ev_transform = self._vehicle.get_transform()
        ev_bbox = self._vehicle.bounding_box
        snap_shot = self._world.get_snapshot()

        def is_within_distance(w):
            ev_loc = ev_transform.location
            c_distance = abs(ev_loc.x - w.location.x) < self._distance_threshold \
                and abs(ev_loc.y - w.location.y) < self._distance_threshold \
                and abs(ev_loc.z - w.location.z) < 8.0
            c_ev = abs(ev_loc.x - w.location.x) < 1.0 and abs(ev_loc.y - w.location.y) < 1.0
            return c_distance and (not c_ev)

        vehicle_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Car)
        walker_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians)

        vehicles = _get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)
        walkers = _get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)

        result = dict(
            ev_transform=ev_transform,
            ev_bbox=ev_bbox,
            vehicles=vehicles,
            walkers=walkers,
        )
        self._full_history.append(result)
        return result

    def reconstruct_bev(self):
        results = []
        for idx in range(len(self._full_history)):
            item = self._full_history[idx]
            image = make_bev_output(
                item, self._road,
                self._width, self._pixels_per_meter, self._pixels_ev_to_bottom, self._world_offset
            )
            results.append({'rendered': image})
        return results


def make_bev_output(
        item, road, width, pixels_per_meter, pixels_ev_to_bottom, world_offset):

    ev_transform = item['ev_transform']
    ev_bbox = item['ev_bbox']

    ev_loc = ev_transform.location
    ev_rot = ev_transform.rotation

    M_warp = _get_warp_transform(ev_loc, ev_rot, width, pixels_per_meter, pixels_ev_to_bottom, world_offset)

    vehicle_mask = _get_mask_from_actor_list(item['vehicles'], M_warp, width, pixels_per_meter, world_offset)
    walker_mask = _get_mask_from_actor_list(item['walkers'], M_warp, width, pixels_per_meter, world_offset)

    road_mask = cv.warpAffine(road, M_warp, (width, width)).astype(np.bool)

    ev_mask = _get_mask_from_actor_list(
        [(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp, width, pixels_per_meter, world_offset)

    image = image_render(road_mask,
               vehicle_mask, walker_mask, ev_mask, width)

    return image
