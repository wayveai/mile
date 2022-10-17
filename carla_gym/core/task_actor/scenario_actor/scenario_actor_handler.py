"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

from carla_gym.core.task_actor.common.task_vehicle import TaskVehicle
import numpy as np
from importlib import import_module


class ScenarioActorHandler(object):
    def __init__(self, client):
        self.scenario_actors = {}
        self.scenario_agents = {}
        self.hero_vehicles = {}

        self._client = client
        self._world = client.get_world()

    def reset(self, task_config, hero_vehicles):
        self.hero_vehicles = hero_vehicles

        actor_config = task_config.get('actors', {})
        route_config = task_config.get('routes', {})

        for sa_id in actor_config:
            # spawn actors
            bp_filter = actor_config[sa_id]['model']
            blueprint = np.random.choice(self._world.get_blueprint_library().filter(bp_filter))
            blueprint.set_attribute('role_name', sa_id)
            spawn_transform = route_config[sa_id][0]
            carla_vehicle = self._world.try_spawn_actor(blueprint, spawn_transform)
            self._world.tick()
            target_transforms = route_config[sa_id][1:]
            self.scenario_actors[sa_id] = TaskVehicle(carla_vehicle, target_transforms)
            # make agents
            module_str, class_str = actor_config[sa_id]['agent_entry_point'].split(':')
            AgentClass = getattr(
                import_module('carla_gym.core.task_actor.scenario_actor.agents.' + module_str),
                class_str)
            self.scenario_agents[sa_id] = AgentClass(self.scenario_actors[sa_id], self.hero_vehicles,
                                                     **actor_config[sa_id].get('agent_kwargs', {}))

    def tick(self):
        for sa_id in self.scenario_actors:
            action = self.scenario_agents[sa_id].get_action()
            self.scenario_actors[sa_id].apply_control(action)
            self.scenario_actors[sa_id].tick()

    def clean(self):
        for sa_id in self.scenario_actors:
            self.scenario_actors[sa_id].clean()
        self.scenario_actors = {}
        self.scenario_agents = {}
        self.hero_vehicles = {}
