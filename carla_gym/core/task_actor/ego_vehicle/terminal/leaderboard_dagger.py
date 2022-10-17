"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

class LeaderboardDagger(object):

    def __init__(self, ego_vehicle, no_collision=True, no_run_rl=True, no_run_stop=True, max_time=300):
        self._ego_vehicle = ego_vehicle

        self._no_collision = no_collision
        self._no_run_rl = no_run_rl
        self._no_run_stop = no_run_stop
        self._max_time = max_time  # in sec

    def get(self, timestamp):

        info_criteria = self._ego_vehicle.info_criteria

        # Done condition 1: blocked
        c_blocked = info_criteria['blocked'] is not None

        # Done condition 2: route_deviation
        c_route_deviation = info_criteria['route_deviation'] is not None

        # Done condition 3: collision
        c_collision = (info_criteria['collision'] is not None) and self._no_collision

        # Done condition 4: running red light
        c_run_rl = (info_criteria['run_red_light'] is not None) and self._no_run_rl

        # Done condition 5: run stop sign
        if info_criteria['run_stop_sign'] is not None and info_criteria['run_stop_sign']['event'] == 'run':
            c_run_stop = True
        else:
            c_run_stop = False
        c_run_stop = c_run_stop and self._no_run_stop

        # Done condition 6: timeout
        timeout = timestamp['relative_simulation_time'] > self._max_time

        done = c_blocked or c_route_deviation or c_collision or c_run_rl or c_run_stop or timeout

        debug_texts = [
            f'dev:{int(c_route_deviation)} blo:{int(c_blocked)} t_out:{int(timeout)}',
            f'col:{int(c_collision)} redl:{int(c_run_rl)} stop:{int(c_run_stop)}'
        ]

        terminal_debug = {
            'traffic_rule_violated': c_collision or c_run_rl or c_run_stop,
            'blocked': c_blocked,
            'route_deviation': c_route_deviation,
            'debug_texts': debug_texts
        }

        terminal_reward = 0.0
        return done, timeout, terminal_reward, terminal_debug
