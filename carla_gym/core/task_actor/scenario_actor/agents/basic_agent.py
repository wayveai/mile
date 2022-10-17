"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import numpy as np
from .utils.local_planner import LocalPlanner
from .utils.misc import is_within_distance_ahead, compute_yaw_difference


class BasicAgent(object):
    def __init__(self, scenario_vehicle, hero_vehicles, target_speed=0.0, max_skip=20, success_dist=5.0):
        self._scenario_vehicle = scenario_vehicle
        self._world = self._scenario_vehicle.vehicle.get_world()
        self._map = self._world.get_map()

        self._dest_transform = scenario_vehicle.dest_transform
        self._success_dist = success_dist
        self._proximity_threshold = 9.5

        self._local_planner = LocalPlanner(target_speed=target_speed)

    def get_action(self):
        transform = self._scenario_vehicle.vehicle.get_transform()

        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter('*vehicle*')
        walkers_list = actor_list.filter('*walker*')
        vehicle_hazard = self._is_vehicle_hazard(transform, self._scenario_vehicle.vehicle.id, vehicle_list)
        pedestrian_ahead = self._is_walker_hazard(transform, walkers_list)

        # check red light
        redlight_ahead = self._scenario_vehicle.vehicle.is_at_traffic_light()
        # target_reached
        target_reached = transform.location.distance(self._dest_transform.location) < self._success_dist

        if vehicle_hazard or pedestrian_ahead or redlight_ahead or target_reached:
            throttle, steer, brake = 0.0, 0.0, 1.0
        else:
            route_plan = self._scenario_vehicle.route_plan
            # ego_vehicle_speed
            velocity = self._scenario_vehicle.vehicle.get_velocity()
            forward_vec = transform.get_forward_vector()
            vel = np.array([velocity.x, velocity.y, velocity.z])
            f_vec = np.array([forward_vec.x, forward_vec.y, forward_vec.z])
            forward_speed = np.dot(vel, f_vec)
            speed = np.linalg.norm(vel)

            throttle, steer, brake = self._local_planner.run_step(route_plan, transform, forward_speed)

        return np.array([throttle, steer, brake], dtype=np.float64)

    def _is_vehicle_hazard(self, ev_transform, ev_id, vehicle_list):
        ego_vehicle_location = ev_transform.location
        ego_vehicle_orientation = ev_transform.rotation.yaw

        for target_vehicle in vehicle_list:
            if target_vehicle.id == ev_id:
                continue

            loc = target_vehicle.get_location()
            ori = target_vehicle.get_transform().rotation.yaw

            if compute_yaw_difference(ego_vehicle_orientation, ori) <= 150 and \
                is_within_distance_ahead(loc, ego_vehicle_location, ego_vehicle_orientation,
                                         self._proximity_threshold, degree=45):
                return True

        return False

    def _is_walker_hazard(self, ev_transform, walkers_list):
        ego_vehicle_location = ev_transform.location

        for walker in walkers_list:
            loc = walker.get_location()
            dist = loc.distance(ego_vehicle_location)
            degree = 162 / (np.clip(dist, 1.5, 10.5)+0.3)
            if self._is_point_on_sidewalk(loc):
                continue

            if is_within_distance_ahead(loc, ego_vehicle_location, ev_transform.rotation.yaw,
                                        self._proximity_threshold, degree=degree):
                return True
        return False

    def _is_point_on_sidewalk(self, loc):
        wp = self._map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Sidewalk)
        if wp is None:
            return False
        else:
            return True
