"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import carla
import weakref
from .navigation.global_route_planner import GlobalRoutePlanner
from .navigation.route_manipulation import location_route_to_gps, downsample_route
import numpy as np
import logging
import copy

from .criteria import blocked, collision, outside_route_lane, route_deviation, run_stop_sign
from .criteria import encounter_light, run_red_light

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TaskVehicle(object):

    def __init__(self, vehicle, target_transforms, spawn_transforms, endless):
        """
        vehicle: carla.Vehicle
        target_transforms: list of carla.Transform
        """
        self.vehicle = vehicle
        world = self.vehicle.get_world()
        self._map = world.get_map()
        self._world = world

        self.criteria_blocked = blocked.Blocked()
        self.criteria_collision = collision.Collision(self.vehicle, world)
        self.criteria_light = run_red_light.RunRedLight(self._map)
        self.criteria_encounter_light = encounter_light.EncounterLight()
        self.criteria_stop = run_stop_sign.RunStopSign(world)
        self.criteria_outside_route_lane = outside_route_lane.OutsideRouteLane(self._map, self.vehicle.get_location())
        self.criteria_route_deviation = route_deviation.RouteDeviation()

        # navigation
        self._route_completed = 0.0
        self._route_length = 0.0

        self._target_transforms = target_transforms  # transforms

        self._planner = GlobalRoutePlanner(self._map, resolution=1.0)

        self._global_route = []
        self._global_plan_gps = []
        self._global_plan_world_coord = []

        self._trace_route_to_global_target()

        self._spawn_transforms = spawn_transforms

        self._endless = endless
        if len(self._target_transforms) == 0:
            while self._route_length < 1000.0:
                self._add_random_target()

        self._last_route_location = self.vehicle.get_location()
        self.collision_px = False

    def _update_leaderboard_plan(self, route_trace):
        plan_gps = location_route_to_gps(route_trace)
        ds_ids = downsample_route(route_trace, 50)

        self._global_plan_gps += [plan_gps[x] for x in ds_ids]
        self._global_plan_world_coord += [(route_trace[x][0].transform.location, route_trace[x][1]) for x in ds_ids]

    def _add_random_target(self):
        if len(self._target_transforms) == 0:
            last_target_loc = self.vehicle.get_location()
            ev_wp = self._map.get_waypoint(last_target_loc)
            next_wp = ev_wp.next(6)[0]
            new_target_transform = next_wp.transform
        else:
            last_target_loc = self._target_transforms[-1].location
            last_road_id = self._map.get_waypoint(last_target_loc).road_id
            new_target_transform = np.random.choice([x[1] for x in self._spawn_transforms if x[0] != last_road_id])

        route_trace = self._planner.trace_route(last_target_loc, new_target_transform.location)
        self._global_route += route_trace
        self._target_transforms.append(new_target_transform)
        self._route_length += self._compute_route_length(route_trace)
        self._update_leaderboard_plan(route_trace)

    def _trace_route_to_global_target(self):
        current_location = self.vehicle.get_location()
        for tt in self._target_transforms:
            next_target_location = tt.location
            route_trace = self._planner.trace_route(current_location, next_target_location)
            self._global_route += route_trace
            self._route_length += self._compute_route_length(route_trace)
            current_location = next_target_location

        self._update_leaderboard_plan(self._global_route)

    @staticmethod
    def _compute_route_length(route):
        length_in_m = 0.0
        for i in range(len(route)-1):
            d = route[i][0].transform.location.distance(route[i+1][0].transform.location)
            length_in_m += d
        return length_in_m

    def _truncate_global_route_till_local_target(self, windows_size=5):
        ev_location = self.vehicle.get_location()
        closest_idx = 0

        for i in range(len(self._global_route)-1):
            if i > windows_size:
                break

            loc0 = self._global_route[i][0].transform.location
            loc1 = self._global_route[i+1][0].transform.location

            wp_dir = loc1 - loc0
            wp_veh = ev_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0:
                closest_idx = i+1

        distance_traveled = self._compute_route_length(self._global_route[:closest_idx+1])
        self._route_completed += distance_traveled

        if closest_idx > 0:
            self._last_route_location = carla.Location(self._global_route[0][0].transform.location)

        self._global_route = self._global_route[closest_idx:]
        return distance_traveled

    def _is_route_completed(self, percentage_threshold=0.99, distance_threshold=10.0):
        # distance_threshold=10.0
        ev_loc = self.vehicle.get_location()

        percentage_route_completed = self._route_completed / self._route_length
        is_completed = percentage_route_completed > percentage_threshold
        is_within_dist = ev_loc.distance(self._target_transforms[-1].location) < distance_threshold

        return is_completed and is_within_dist

    def tick(self, timestamp):
        distance_traveled = self._truncate_global_route_till_local_target()
        route_completed = self._is_route_completed()
        if self._endless and (len(self._global_route) < 10 or route_completed):
            self._add_random_target()
            route_completed = False

        info_blocked = self.criteria_blocked.tick(self.vehicle, timestamp)
        info_collision = self.criteria_collision.tick(self.vehicle, timestamp)
        info_light = self.criteria_light.tick(self.vehicle, timestamp)
        info_encounter_light = self.criteria_encounter_light.tick(self.vehicle, timestamp)
        info_stop = self.criteria_stop.tick(self.vehicle, timestamp)
        info_outside_route_lane = self.criteria_outside_route_lane.tick(self.vehicle, timestamp, distance_traveled)
        info_route_deviation = self.criteria_route_deviation.tick(
            self.vehicle, timestamp, self._global_route[0][0], distance_traveled, self._route_length)

        info_route_completion = {
            'step': timestamp['step'],
            'simulation_time': timestamp['relative_simulation_time'],
            'route_completed_in_m': self._route_completed,
            'route_length_in_m': self._route_length,
            'is_route_completed': route_completed
        }

        self._info_criteria = {
            'route_completion': info_route_completion,
            'outside_route_lane': info_outside_route_lane,
            'route_deviation': info_route_deviation,
            'blocked': info_blocked,
            'collision': info_collision,
            'run_red_light': info_light,
            'encounter_light': info_encounter_light,
            'run_stop_sign': info_stop
        }

        # turn on light
        weather = self._world.get_weather()
        if weather.sun_altitude_angle < 0.0:
            vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        else:
            vehicle_lights = carla.VehicleLightState.NONE
        self.vehicle.set_light_state(carla.VehicleLightState(vehicle_lights))

        return self._info_criteria

    def clean(self):
        self.criteria_collision.clean()
        self.vehicle.destroy()

    @property
    def info_criteria(self):
        return self._info_criteria

    @property
    def dest_transform(self):
        return self._target_transforms[-1]

    @property
    def route_plan(self):
        return self._global_route

    @property
    def global_plan_gps(self):
        return self._global_plan_gps

    @property
    def global_plan_world_coord(self):
        return self._global_plan_world_coord

    @property
    def route_length(self):
        return self._route_length

    @property
    def route_completed(self):
        return self._route_completed

    def get_route_transform(self):
        loc0 = self._last_route_location
        loc1 = self._global_route[0][0].transform.location

        if loc1.distance(loc0) < 0.1:
            yaw = self._global_route[0][0].transform.rotation.yaw
        else:
            f_vec = loc1 - loc0
            yaw = np.rad2deg(np.arctan2(f_vec.y, f_vec.x))
        rot = carla.Rotation(yaw=yaw)
        return carla.Transform(location=loc0, rotation=rot)
