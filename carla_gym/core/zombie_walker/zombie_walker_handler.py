"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import carla
import numpy as np
import logging
from .zombie_walker import ZombieWalker


class ZombieWalkerHandler(object):

    def __init__(self, client, spawn_distance_to_ev=10.0):
        self._logger = logging.getLogger(__name__)
        self.zombie_walkers = {}
        self._client = client
        self._world = client.get_world()
        self._spawn_distance_to_ev = spawn_distance_to_ev

    def reset(self, num_zombie_walkers, ev_spawn_locations):
        if type(num_zombie_walkers) is list:
            n_spawn = np.random.randint(num_zombie_walkers[0], num_zombie_walkers[1])
        else:
            n_spawn = num_zombie_walkers
        self._spawn(n_spawn, ev_spawn_locations)
        self._logger.debug(f'Spawned {len(self.zombie_walkers)} zombie walkers. '
                           f'Should Spawn {num_zombie_walkers}')

    def _spawn(self, num_zombie_walkers, ev_spawn_locations, max_trial=10, tick=True):
        SpawnActor = carla.command.SpawnActor
        walker_bp_library = self._world.get_blueprint_library().filter('walker.pedestrian.*')
        walker_controller_bp = self._world.get_blueprint_library().find('controller.ai.walker')

        def proximity_to_ev(location): return any([ev_loc.distance(location) < self._spawn_distance_to_ev
                                                   for ev_loc in ev_spawn_locations])

        controller_ids = []
        walker_ids = []
        num_spawned = 0
        n_trial = 0
        while num_spawned < num_zombie_walkers:
            spawn_points = []
            _walkers = []
            _controllers = []

            for i in range(num_zombie_walkers - num_spawned):
                is_proximity_to_ev = True
                spawn_loc = None
                while is_proximity_to_ev:
                    spawn_loc = self._world.get_random_location_from_navigation()
                    if spawn_loc is not None:
                        is_proximity_to_ev = proximity_to_ev(spawn_loc)
                spawn_points.append(carla.Transform(location=spawn_loc))

            batch = []
            for spawn_point in spawn_points:
                walker_bp = np.random.choice(walker_bp_library)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                batch.append(SpawnActor(walker_bp, spawn_point))

            for result in self._client.apply_batch_sync(batch, tick):
                if not result.error:
                    num_spawned += 1
                    _walkers.append(result.actor_id)

            batch = [SpawnActor(walker_controller_bp, carla.Transform(), walker) for walker in _walkers]
            for result in self._client.apply_batch_sync(batch, tick):
                if result.error:
                    self._logger.error(result.error)
                else:
                    _controllers.append(result.actor_id)

            controller_ids.extend(_controllers)
            walker_ids.extend(_walkers)

            n_trial += 1
            if n_trial == max_trial and (num_zombie_walkers - num_spawned)>0:
                self._logger.warning(f'{self._world.get_map().name}: '
                                     f'Spawning zombie walkers max trial {n_trial} reached! '
                                     f'spawned/to_spawn: {num_spawned}/{num_zombie_walkers}')
                break

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        # self._world.tick()

        for w_id, c_id in zip(walker_ids, controller_ids):
            self.zombie_walkers[w_id] = ZombieWalker(w_id, c_id, self._world)

        return self.zombie_walkers

    def tick(self):
        pass

    def clean(self):
        live_walkers_list = [walker.id for walker in self._world.get_actors().filter("*walker.pedestrian*")]

        for zw_id, zw in self.zombie_walkers.items():
            if zw_id in live_walkers_list:
                zw.clean()

        self.zombie_walkers = {}
