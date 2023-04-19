"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import cv2 as cv
from pathlib import Path
import h5py

from utils.profiling_utils import profile

from collections import deque
import carla_gym.utils.transforms as trans_utils
import carla
from carla_gym.core.task_actor.common.navigation.global_route_planner import GlobalRoutePlanner
from carla_gym.core.task_actor.common.navigation.route_manipulation import location_route_to_gps, downsample_route
import numpy as np
import logging

from carla_gym.core.task_actor.common.criteria import blocked, collision, outside_route_lane, route_deviation, run_stop_sign
from carla_gym.core.task_actor.common.criteria import encounter_light, run_red_light

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _compute_route_length(route):
    length_in_m = 0.0
    for i in range(len(route) - 1):
        d = route[i][0].transform.location.distance(route[i + 1][0].transform.location)
        length_in_m += d
    return length_in_m

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


def make_route_mask(M_warp, route_plan, width, pixels_per_meter, world_offset):
    route_mask = np.zeros([width, width], dtype=np.uint8)
    route_in_pixel = np.array([[_world_to_pixel(wp.transform.location, pixels_per_meter, world_offset)]
                               for wp, _ in route_plan[0:80]])
    route_warped = cv.transform(route_in_pixel, M_warp)
    cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
    route_mask = route_mask.astype(np.bool)
    return route_mask


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
    vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []

    for idx in history_idx:
        idx = max(idx, -1 * qsize)

        vehicles, walkers, tl_green, tl_yellow, tl_red, stops = history_queue[idx]

        vehicle_masks.append(
            _get_mask_from_actor_list(vehicles, M_warp, width, pixels_per_meter, world_offset))
        walker_masks.append(
            _get_mask_from_actor_list(walkers, M_warp, width, pixels_per_meter, world_offset))
        tl_green_masks.append(
            _get_mask_from_stopline_vtx(width, pixels_per_meter, world_offset, tl_green, M_warp))
        tl_yellow_masks.append(
            _get_mask_from_stopline_vtx(width, pixels_per_meter, world_offset, tl_yellow, M_warp))
        tl_red_masks.append(
            _get_mask_from_stopline_vtx(width, pixels_per_meter, world_offset, tl_red, M_warp))
        stop_masks.append(
            _get_mask_from_actor_list(stops, M_warp, width, pixels_per_meter, world_offset))

    return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks


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


def _get_masks(road_mask, route_mask, lane_mask_all, lane_mask_broken,
               tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks,
               vehicle_masks, walker_masks, width, history_idx):
    c_road = road_mask * 255
    c_route = route_mask * 255
    c_lane = lane_mask_all * 255
    c_lane[lane_mask_broken] = 120

    # masks with history
    c_tl_history = []
    for i in range(len(history_idx)):
        c_tl = np.zeros([width, width], dtype=np.uint8)
        c_tl[tl_green_masks[i]] = 80
        c_tl[tl_yellow_masks[i]] = 170
        c_tl[tl_red_masks[i]] = 255
        c_tl[stop_masks[i]] = 255
        c_tl_history.append(c_tl)

    c_vehicle_history = [m * 255 for m in vehicle_masks]
    c_walker_history = [m * 255 for m in walker_masks]

    masks = np.stack((c_road, c_route, c_lane, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)
    masks = np.transpose(masks, [2, 0, 1])
    return masks


def image_render(road_mask, route_mask, lane_mask_all, lane_mask_broken,
               tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks,
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
    for i, mask in enumerate(tl_green_masks):
        image[mask] = tint(COLOR_GREEN, (h_len-i)*0.2)
    for i, mask in enumerate(tl_yellow_masks):
        image[mask] = tint(COLOR_YELLOW, (h_len-i)*0.2)
    for i, mask in enumerate(tl_red_masks):
        image[mask] = tint(COLOR_RED, (h_len-i)*0.2)

    for i, mask in enumerate(vehicle_masks):
        image[mask] = tint(COLOR_BLUE, (h_len-i)*0.2)
    for i, mask in enumerate(walker_masks):
        image[mask] = tint(COLOR_CYAN, (h_len-i)*0.2)

    image[ev_mask] = COLOR_WHITE
    # image[obstacle_mask] = COLOR_BLUE
    return image



class TrafficLightHandlerInstance:
    def __init__(self):
        self.num_tl = 0
        self.list_tl_actor = []
        self.list_tv_loc = []
        self.list_stopline_wps = []
        self.list_stopline_vtx = []
        self.list_junction_paths = []
        self.carla_map = None


def init_tl_instance(traffic_light_handler, world):
    traffic_light_handler.carla_map = world.get_map()

    traffic_light_handler.num_tl = 0
    traffic_light_handler.list_tl_actor = []
    traffic_light_handler.list_tv_loc = []
    traffic_light_handler.list_stopline_wps = []
    traffic_light_handler.list_stopline_vtx = []
    traffic_light_handler.list_junction_paths = []

    all_actors = world.get_actors()
    for _actor in all_actors:
        if 'traffic_light' in _actor.type_id:
            tv_loc, stopline_wps, stopline_vtx, junction_paths = _get_traffic_light_waypoints(
                _actor, traffic_light_handler.carla_map)

            traffic_light_handler.list_tl_actor.append(_actor)
            traffic_light_handler.list_tv_loc.append(tv_loc)
            traffic_light_handler.list_stopline_wps.append(stopline_wps)
            traffic_light_handler.list_stopline_vtx.append(stopline_vtx)
            traffic_light_handler.list_junction_paths.append(junction_paths)

            traffic_light_handler.num_tl += 1
    return traffic_light_handler


def tl_get_stopline_vtx(traffic_light_handler, veh_loc, color, dist_threshold=50.0):
    if color == 0:
        tl_state = carla.TrafficLightState.Green
    elif color == 1:
        tl_state = carla.TrafficLightState.Yellow
    elif color == 2:
        tl_state = carla.TrafficLightState.Red
    else:
        raise RuntimeError('unknown color')

    stopline_vtx = []
    for i in range(traffic_light_handler.num_tl):
        traffic_light = traffic_light_handler.list_tl_actor[i]
        tv_loc = traffic_light_handler.list_tv_loc[i]
        if tv_loc.distance(veh_loc) > dist_threshold:
            continue
        if traffic_light.state != tl_state:
            continue
        stopline_vtx += traffic_light_handler.list_stopline_vtx[i]

    return stopline_vtx


def _get_traffic_light_waypoints(traffic_light, carla_map):
    """
    get area of a given traffic light
    adapted from "carla-simulator/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py"
    """
    base_transform = traffic_light.get_transform()
    tv_loc = traffic_light.trigger_volume.location
    tv_ext = traffic_light.trigger_volume.extent

    # Discretize the trigger box into points
    x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes
    area = []
    for x in x_values:
        point_location = base_transform.transform(tv_loc + carla.Location(x=x))
        area.append(point_location)

    # Get the waypoints of these points, removing duplicates
    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
            ini_wps.append(wpx)

    # Leaderboard: Advance them until the intersection
    stopline_wps = []
    stopline_vertices = []
    junction_wps = []
    for wpx in ini_wps:
        # Below: just use trigger volume, otherwise it's on the zebra lines.
        # stopline_wps.append(wpx)
        # vec_forward = wpx.transform.get_forward_vector()
        # vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

        # loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
        # loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
        # stopline_vertices.append([loc_left, loc_right])

        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            if next_wp and not next_wp.is_intersection:
                wpx = next_wp
            else:
                break
        junction_wps.append(wpx)

        stopline_wps.append(wpx)
        vec_forward = wpx.transform.get_forward_vector()
        vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

        loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
        loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
        stopline_vertices.append([loc_left, loc_right])

    # all paths at junction for this traffic light
    junction_paths = []
    path_wps = []
    wp_queue = deque(junction_wps)
    while len(wp_queue) > 0:
        current_wp = wp_queue.pop()
        path_wps.append(current_wp)
        next_wps = current_wp.next(1.0)
        for next_wp in next_wps:
            if next_wp.is_junction:
                wp_queue.append(next_wp)
            else:
                junction_paths.append(path_wps)
                path_wps = []

    return carla.Location(base_transform.transform(tv_loc)), stopline_wps, stopline_vertices, junction_paths

def _make_random_plan(initial_location, world_map, spawn_transforms):
    planner = GlobalRoutePlanner(world_map, resolution=1.0)
    route_length = 0.0
    target_transforms = []
    global_route = []
    while route_length < 1000.0:
        if len(target_transforms) == 0:
            last_target_loc = initial_location
            ev_wp = world_map.get_waypoint(last_target_loc)
            next_wp = ev_wp.next(6)[0]
            new_target_transform = next_wp.transform
        else:
            last_target_loc = target_transforms[-1].location
            last_road_id = world_map.get_waypoint(last_target_loc).road_id
            new_target_transform = np.random.choice([x[1] for x in spawn_transforms if x[0] != last_road_id])

        route_trace = planner.trace_route(last_target_loc, new_target_transform.location)
        global_route += route_trace
        target_transforms.append(new_target_transform)
        route_length += _compute_route_length(route_trace)
    return global_route


class VehicleWrapper(object):
    def __init__(self, vehicle, spawn_transforms):
        self.vehicle = vehicle
        self.route_plan = _make_random_plan(
            self.vehicle.get_location(), self.vehicle.get_world().get_map(), spawn_transforms)


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


class VectorizedInputManager:
    def __init__(self, obs_configs):
        self._width = int(obs_configs['width_in_pixels'])
        self._pixels_ev_to_bottom = obs_configs['pixels_ev_to_bottom']
        self._pixels_per_meter = obs_configs['pixels_per_meter']
        self._history_idx = obs_configs['history_idx']
        self._scale_mask_col = obs_configs.get('scale_mask_col', 1.1)
        self._obs_config = obs_configs

        self._history_queue = deque(maxlen=20)

        self._image_channels = 3
        self._masks_channels = 3 + 3*len(self._history_idx)
        self._vehicle_wrapper = None
        self._world = None

        self._map_dir = Path(__file__).resolve().parent / 'carla_gym/core/obs_manager/birdview/maps'

    def attach_ego_vehicle(self, parent_actor):
        self._vehicle_wrapper = parent_actor
        self._world = self._vehicle_wrapper.vehicle.get_world()

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

    def get_observation(self, tl_manager):
        ev_transform = self._vehicle_wrapper.vehicle.get_transform()
        ev_loc = ev_transform.location
        ev_rot = ev_transform.rotation
        ev_bbox = self._vehicle_wrapper.vehicle.bounding_box
        snap_shot = self._world.get_snapshot()

        def is_within_distance(w):
            c_distance = abs(ev_loc.x - w.location.x) < self._distance_threshold \
                and abs(ev_loc.y - w.location.y) < self._distance_threshold \
                and abs(ev_loc.z - w.location.z) < 8.0
            c_ev = abs(ev_loc.x - w.location.x) < 1.0 and abs(ev_loc.y - w.location.y) < 1.0
            return c_distance and (not c_ev)

        vehicle_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Car)
        walker_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Pedestrians)

        vehicles = _get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)
        walkers = _get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)

        tl_green = tl_get_stopline_vtx(tl_manager, ev_loc, 0)
        tl_yellow = tl_get_stopline_vtx(tl_manager, ev_loc, 1)
        tl_red = tl_get_stopline_vtx(tl_manager, ev_loc, 2)
        stops = [] # _get_stops(self._vehicle_wrapper.criteria_stop)  (stop has tricky logic, let's ignore for now)

        self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops))

        M_warp = _get_warp_transform(ev_loc, ev_rot, self._width, self._pixels_per_meter, self._pixels_ev_to_bottom, self._world_offset)

        # objects with history
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
            = _get_history_masks(
            M_warp, self._history_queue, self._history_idx, self._width, self._pixels_per_meter, self._world_offset)

        road_mask, lane_mask_all, lane_mask_broken = make_lane_masks(
            M_warp, self._road, self._width, self._lane_marking_all, self._lane_marking_white_broken)

        # route_mask
        route_mask = make_route_mask(
            M_warp, self._vehicle_wrapper.route_plan, self._width, self._pixels_per_meter, self._world_offset)

        # ev_mask
        ev_mask = _get_mask_from_actor_list(
            [(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp, self._width, self._pixels_per_meter, self._world_offset)

        # render
        image = image_render(road_mask, route_mask, lane_mask_all, lane_mask_broken,
                   tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks,
                   vehicle_masks, walker_masks, ev_mask, self._width, self._history_idx)

        masks  = _get_masks(
           road_mask, route_mask, lane_mask_all, lane_mask_broken,
           tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks,
           vehicle_masks, walker_masks, self._width, self._history_idx)

        result = {'rendered': image, 'masks': masks}

        return result
