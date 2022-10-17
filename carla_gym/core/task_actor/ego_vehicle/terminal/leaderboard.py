"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

class Leaderboard(object):

    def __init__(self, ego_vehicle, max_time=None):
        self._ego_vehicle = ego_vehicle
        self._max_time = max_time # in sec

    def get(self, timestamp):

        info_criteria = self._ego_vehicle.info_criteria
        # Done condition 1: route completed
        c_route = info_criteria['route_completion']['is_route_completed']

        # Done condition 2: blocked
        c_blocked = info_criteria['blocked'] is not None

        # Done condition 3: route_deviation
        c_route_deviation = info_criteria['route_deviation'] is not None

        # Done condition 4: timeout
        if self._max_time is not None:
            timeout = timestamp['relative_simulation_time'] > self._max_time
        else:
            timeout = False

        done = c_route or c_blocked or c_route_deviation or timeout
        
        debug_texts = [
            f'cpl:{int(c_route)} dev:{int(c_route_deviation)} blo:{int(c_blocked)} t_out:{int(timeout)}'
        ]

        terminal_debug = {
            'blocked': c_blocked,
            'route_deviation': c_route_deviation,
            'debug_texts': debug_texts
        }

        terminal_reward = 0.0
        return done, timeout, terminal_reward, terminal_debug
