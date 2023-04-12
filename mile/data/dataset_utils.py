import torch
import numpy as np

import carla
import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util
from agents.navigation.local_planner import RoadOption


def binary_to_integer(binary_array, n_bits):
    """
    Parameters
    ----------
        binary_array: shape (n, n_bits)

    Returns
    -------
        integer_array: shape (n,) np.int32
    """
    return (binary_array @ 2 ** np.arange(n_bits, dtype=binary_array.dtype)).astype(np.int32)


def integer_to_binary(integer_array, n_bits):
    """
    Parameters
    ----------
        integer_array: np.ndarray<int32> (n,)
        n_bits: int

    Returns
    -------
        binary_array: np.ndarray<float32> (n, n_bits)

    """
    return (((integer_array[:, None] & (1 << np.arange(n_bits)))) > 0).astype(np.float32)


def calculate_birdview_labels(birdview, n_classes, has_time_dimension=False):
    """
    Parameters
    ----------
        birdview: torch.Tensor<float> (C, H, W)
        n_classes: int
            number of total classes
        has_time_dimension: bool

    Returns
    -------
        birdview_label: (H, W)
    """
    # When a pixel contains two labels, argmax will output the first one that is encountered.
    # By reversing the order, we prioritise traffic lights over road.
    dim = 0
    if has_time_dimension:
        dim = 1
    birdview_label = torch.argmax(birdview.flip(dims=[dim]), dim=dim)
    # We then re-normalise the classes in the normal order.
    birdview_label = (n_classes - 1) - birdview_label
    return birdview_label


def preprocess_measurements(route_command, ego_gps, target_gps, imu):
    # preprocess measurements
    if route_command == RoadOption.VOID.value:
        route_command = RoadOption.LANEFOLLOW.value
    route_command -= 1
    route_command = np.array(route_command, dtype=np.int64)

    loc_in_ev = preprocess_gps(ego_gps, target_gps, imu)
    gps_vector = np.array([loc_in_ev.x, loc_in_ev.y], dtype=np.float32)
    return route_command, gps_vector


def preprocess_gps(ego_gps, target_gps, imu):
    # imu nan bug
    compass = 0.0 if np.isnan(imu[-1]) else imu[-1]
    target_vec_in_global = gps_util.gps_to_location(target_gps) - gps_util.gps_to_location(ego_gps)
    ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
    loc_in_ev = trans_utils.vec_global_to_ref(target_vec_in_global, ref_rot_in_global)
    return loc_in_ev


def preprocess_birdview_and_routemap(birdview):
    ROUTE_MAP_INDEX = 1
    # road, lane markings, vehicles, pedestrians
    relevant_indices = [0, 2, 6, 10]

    if isinstance(birdview, np.ndarray):
        birdview = torch.from_numpy(birdview)
    has_time_dimension = True
    if len(birdview.shape) == 3:
        birdview = birdview.unsqueeze(0)
        has_time_dimension = False
    # Birdview has values in {0, 255}. Convert to {0, 1}

    # lights and stops
    light_and_stop_channel = birdview[:, -1:]
    green_light = (light_and_stop_channel == 80).float()
    yellow_light = (light_and_stop_channel == 170).float()
    red_light_and_stop = (light_and_stop_channel == 255).float()

    remaining = birdview[:, relevant_indices]
    remaining[remaining > 0] = 1
    remaining = remaining.float()

    # Traffic light and stop.
    processed_birdview = torch.cat([remaining, green_light, yellow_light, red_light_and_stop], dim=1)
    # background
    tmp = processed_birdview.sum(dim=1, keepdim=True)
    background = (tmp == 0).float()

    processed_birdview = torch.cat([background, processed_birdview], dim=1)

    # Route map
    route_map = birdview[:, ROUTE_MAP_INDEX]
    route_map[route_map > 0] = 255
    route_map = route_map.to(torch.uint8)

    if not has_time_dimension:
        processed_birdview = processed_birdview[0]
        route_map = route_map[0]
    return processed_birdview, route_map
