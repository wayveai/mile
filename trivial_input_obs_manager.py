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


def make_lane_masks(M_warp, road, width, lane_marking_all, lane_marking_white_broken):
    # road_mask, lane_mask
    road_mask = cv.warpAffine(road, M_warp, (width, width)).astype(np.bool)
    lane_mask_all = cv.warpAffine(lane_marking_all, M_warp, (width, width)).astype(np.bool)
    lane_mask_broken = cv.warpAffine(lane_marking_white_broken, M_warp,
                                     (width, width)).astype(np.bool)
    return road_mask, lane_mask_all, lane_mask_broken


def _get_mask_from_stopline_vtx(width, pixels_per_meter, world_offset, stopline_vtx, M_warp):
    mask = np.zeros([width, width], dtype=np.uint8)
    for sp_locs in stopline_vtx:
        stopline_in_pixel = np.array([[_world_to_pixel(x, pixels_per_meter, world_offset)] for x in sp_locs])
        stopline_warped = cv.transform(stopline_in_pixel, M_warp)
        cv.line(mask, tuple(stopline_warped[0, 0]), tuple(stopline_warped[1, 0]),
                color=1, thickness=6)
    return mask.astype(np.bool)


def _get_history_masks(M_warp, history_queue, history_idx, width, pixels_per_meter, world_offset):
    qsize = len(history_queue)
    vehicle_masks, walker_masks, stop_masks = [], [], []

    for idx in history_idx:
        idx = max(idx, -1 * qsize)

        item = history_queue[idx]

        vehicle_masks.append(
            _get_mask_from_actor_list(item['vehicles'], M_warp, width, pixels_per_meter, world_offset))
        walker_masks.append(
            _get_mask_from_actor_list(item['walkers'], M_warp, width, pixels_per_meter, world_offset))
        stop_masks.append(
            _get_mask_from_actor_list(item['stops'], M_warp, width, pixels_per_meter, world_offset))

    return vehicle_masks, walker_masks, stop_masks


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


def image_render(road_mask, route_mask, lane_mask_all, lane_mask_broken,
               stop_masks,
               vehicle_masks, walker_masks, ev_mask, width, history_idx):
    # render
    image = np.zeros([width, width, 3], dtype=np.uint8)
    image[road_mask] = COLOR_ALUMINIUM_5
    image[route_mask] = COLOR_ALUMINIUM_3
    image[lane_mask_all] = COLOR_MAGENTA
    image[lane_mask_broken] = COLOR_MAGENTA_2

    h_len = len(history_idx)-1
    for i, mask in enumerate(stop_masks):
        image[mask] = tint(COLOR_YELLOW_2, (h_len-i)*0.2)

    for i, mask in enumerate(vehicle_masks):
        image[mask] = tint(COLOR_BLUE, (h_len-i)*0.2)
    for i, mask in enumerate(walker_masks):
        image[mask] = tint(COLOR_CYAN, (h_len-i)*0.2)

    image[ev_mask] = COLOR_WHITE
    return image


COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
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


def _get_stops(criteria_stop):
    stop_sign = criteria_stop._target_stop_sign
    stops = []
    if (stop_sign is not None) and (not criteria_stop._stop_completed):
        bb_loc = carla.Location(stop_sign.trigger_volume.location)
        bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
        bb_ext.x = max(bb_ext.x, bb_ext.y)
        bb_ext.y = max(bb_ext.x, bb_ext.y)
        trans = stop_sign.get_transform()
        stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]
    return stops


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
        self._masks_channels = 3 + 3*len(self._history_idx)
        self._vehicle = None
        self._world = None

        self._map_dir = Path(__file__).resolve().parent / 'carla_gym/core/obs_manager/birdview/maps'

    def attach_ego_vehicle(self, parent_actor):
        self._vehicle = parent_actor
        self._world = self._vehicle.get_world()

        maps_h5_path = self._map_dir / (self._world.get_map().name.rsplit('/', 1)[-1] + '.h5')
        with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
            self._road = np.array(hf['road'], dtype=np.uint8)
            self._lane_marking_all = np.array(hf['lane_marking_all'], dtype=np.uint8)
            self._lane_marking_white_broken = np.array(hf['lane_marking_white_broken'], dtype=np.uint8)
            # self._shoulder = np.array(hf['shoulder'], dtype=np.uint8)
            # self._parking = np.array(hf['parking'], dtype=np.uint8)
            # self._sidewalk = np.array(hf['sidewalk'], dtype=np.uint8)
            # self._lane_marking_yellow_broken = np.array(hf['lane_marking_yellow_broken'], dtype=np.uint8)
            # self._lane_marking_yellow_solid = np.array(hf['lane_marking_yellow_solid'], dtype=np.uint8)
            # self._lane_marking_white_solid = np.array(hf['lane_marking_white_solid'], dtype=np.uint8)

            self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
            assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))

        self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)
        # dilate road mask, lbc draw road polygon with 10px boarder
        # kernel = np.ones((11, 11), np.uint8)
        # self._road = cv.dilate(self._road, kernel, iterations=1)

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

        stops = []

        result = dict(
            ev_transform=ev_transform,
            ev_bbox=ev_bbox,
            vehicles=vehicles,
            walkers=walkers,
            stops=stops
        )
        self._full_history.append(result)
        return result

    def reconstruct_bev(self):
        results = []
        for idx in range(len(self._full_history)):
            local_history = self._full_history[:idx+1]
            image = make_bev_output(
                local_history, self._history_idx,
                self._road, self._lane_marking_all, self._lane_marking_white_broken,
                self._width, self._pixels_per_meter, self._pixels_ev_to_bottom, self._world_offset
            )
            results.append({'rendered': image})
        return results


def make_bev_output(
        history_queue, history_idx,
        road, lane_marking_all, lane_marking_white_broken,
        width, pixels_per_meter, pixels_ev_to_bottom, world_offset):

    latest = history_queue[-1]
    ev_transform = latest['ev_transform']
    ev_bbox = latest['ev_bbox']

    ev_loc = ev_transform.location
    ev_rot = ev_transform.rotation

    M_warp = _get_warp_transform(ev_loc, ev_rot, width, pixels_per_meter, pixels_ev_to_bottom, world_offset)

    # objects with history
    vehicle_masks, walker_masks, stop_masks \
        = _get_history_masks(
        M_warp, history_queue, history_idx, width, pixels_per_meter, world_offset)

    road_mask, lane_mask_all, lane_mask_broken = make_lane_masks(
        M_warp, road, width, lane_marking_all, lane_marking_white_broken)

    # route_mask
    route_mask = np.zeros([width, width], dtype=np.uint8)

    # ev_mask
    ev_mask = _get_mask_from_actor_list(
        [(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp, width, pixels_per_meter, world_offset)

    # render
    image = image_render(road_mask, route_mask, lane_mask_all, lane_mask_broken,
               stop_masks,
               vehicle_masks, walker_masks, ev_mask, width, history_idx)

    return image
