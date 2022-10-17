"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

from pathlib import Path
from gym.envs.registration import register

CARLA_GYM_ROOT_DIR = Path(__file__).resolve().parent

# Declare available environments with a brief description
_AVAILABLE_ENVS = {
    'Endless-v0': {
        'entry_point': 'carla_gym.envs:EndlessEnv',
        'description': 'endless env for rl training and testing',
        'kwargs': {}
    },
    'LeaderBoard-v0': {
        'entry_point': 'carla_gym.envs:LeaderboardEnv',
        'description': 'leaderboard route with no-that-dense backtround traffic',
        'kwargs': {}
    }
}


for env_id, val in _AVAILABLE_ENVS.items():
    register(id=env_id, entry_point=val.get('entry_point'), kwargs=val.get('kwargs'))


def list_available_envs():
    print('Environment-ID: Short-description')
    import pprint
    available_envs = {}
    for env_id, val in _AVAILABLE_ENVS.items():
        available_envs[env_id] = val.get('description')
    pprint.pprint(available_envs)
