import numpy as np
import math
import torch
import carla
from mile.constants import EARTH_RADIUS_EQUA


def bev_params_to_intrinsics(size, scale, offsetx):
    """
        size: number of pixels (width, height)
        scale: pixel size (in meters)
        offsetx: offset in x direction (direction of car travel)
    """
    intrinsics_bev = np.array([
        [1/scale, 0, size[0]/2 + offsetx],
        [0, -1/scale, size[1]/2],
        [0, 0, 1]
    ], dtype=np.float32)
    return intrinsics_bev

def intrinsics_inverse(intrinsics):
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    one = torch.ones_like(fx)
    zero = torch.zeros_like(fx)
    intrinsics_inv = torch.stack((
        torch.stack((1/fx, zero, -cx/fx), -1),
        torch.stack((zero, 1/fy, -cy/fy), -1),
        torch.stack((zero, zero, one), -1),
    ), -2)
    return intrinsics_inv


def get_out_of_view_mask(cfg):
    """ Returns a mask of everything that is not visible from the image given a certain bird's-eye view grid."""
    fov = cfg.IMAGE.FOV
    w = cfg.IMAGE.SIZE[1]
    resolution = cfg.BEV.RESOLUTION

    f = w / (2 * np.tan(fov * np.pi / 360.0))
    c_u = w / 2 - cfg.IMAGE.CROP[0]  # Adjust center point due to cropping

    bev_left = -np.round((cfg.BEV.SIZE[0] // 2) * resolution, decimals=1)
    bev_right = np.round((cfg.BEV.SIZE[0] // 2) * resolution, decimals=1)
    bev_bottom = 0.01
    # The camera is not exactly at the bottom of the bev image, so need to offset it.
    camera_offset = (cfg.BEV.SIZE[1] / 2 + cfg.BEV.OFFSET_FORWARD) * resolution + cfg.IMAGE.CAMERA_POSITION[0]
    bev_top = np.round(cfg.BEV.SIZE[1] * resolution - camera_offset, decimals=1)

    x, z = np.arange(bev_left, bev_right, resolution), np.arange(bev_bottom, bev_top, resolution)
    ucoords = x / z[:, None] * f + c_u

    # Return all points which lie within the camera bounds
    new_w = cfg.IMAGE.CROP[2] - cfg.IMAGE.CROP[0]
    mask = (ucoords >= 0) & (ucoords < new_w)
    mask = ~mask[::-1]
    mask_behind_ego_vehicle = np.ones((int(camera_offset / resolution), mask.shape[1]), dtype=np.bool)
    return np.vstack([mask, mask_behind_ego_vehicle])


def calculate_geometry(image_fov, height, width, forward, right, up, roll, pitch, yaw):
    """Intrinsics and extrinsics for a single camera.
    See https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/leaderboard/camera.py
    and https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/recording/sensors/camera.py
    """
    f = width / (2 * np.tan(image_fov * np.pi / 360.0))
    cx = width / 2
    cy = height / 2
    intrinsics = np.float32([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    extrinsics = get_extrinsics(forward, right, up, yaw, pitch, roll)
    return intrinsics, extrinsics


def get_extrinsics(forward, right, up, yaw, pitch, roll):
    # After multiplying by the extrinsics, we want the axis to be (forward, left, up), and centered in the inertial center of the ego-vehicle.
    assert yaw == pitch == roll == 0.0
    return np.float32(
        [
            [0, 0, 1, forward],
            [-1, 0, 0, -right],
            [0, -1, 0, up],
            [0, 0, 0, 1],
        ]
    )


def gps_to_location(gps):
    lat, lon, z = gps
    lat = float(lat)
    lon = float(lon)
    z = float(z)
    location = carla.Location(z=z)
    location.x = lon / 180.0 * (math.pi * EARTH_RADIUS_EQUA)
    location.y = -1.0 * math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * EARTH_RADIUS_EQUA
    return location


def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
    """
    :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
    :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
    :return: carla.Vector3D in ref coordinate
    """
    R = carla_rot_to_mat(ref_rot_in_global)
    np_vec_in_global = np.array(
        [[target_vec_in_global.x], [target_vec_in_global.y], [target_vec_in_global.z]]
    )
    np_vec_in_ref = R.T.dot(np_vec_in_global)
    target_vec_in_ref = carla.Vector3D(
        x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0]
    )
    return target_vec_in_ref


def carla_rot_to_mat(carla_rotation):
    """
    Transform rpy in carla.Rotation to rotation matrix in np.array
    :param carla_rotation: carla.Rotation
    :return: np.array rotation matrix
    """
    roll = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw = np.deg2rad(carla_rotation.yaw)
    yaw_matrix = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    roll_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)],
        ]
    )
    return yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
