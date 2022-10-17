"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import gym
import json
from pathlib import Path
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os.path
import sys
from time import time

from gym.wrappers.monitoring.video_recorder import ImageEncoder
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from carla_gym.utils import config_utils
from utils import server_utils
from agents.rl_birdview.utils.wandb_callback import WandbCallback
from mile.constants import CARLA_FPS

log = logging.getLogger(__name__)


def run_single(run_name, env, agents_dict, agents_log_dir, log_video, max_step=None):
    list_render = []
    ep_stat_dict = {}
    ep_event_dict = {}
    for actor_id, agent in agents_dict.items():
        log_dir = agents_log_dir / actor_id
        log_dir.mkdir(parents=True, exist_ok=True)
        agent.reset(log_dir / f'{run_name}.log')

    log.info(f'Start Benchmarking {run_name}.')
    obs = env.reset()
    timestamp = env.timestamp
    done = {'__all__': False}

    counter = 0
    warm_start = 25

    while not done['__all__']:
        control_dict = {}
        for actor_id, agent in agents_dict.items():
            control_dict[actor_id] = agent.run_step(obs[actor_id], timestamp)

        obs, reward, done, info = env.step(control_dict)

        render_imgs = []
        for actor_id, agent in agents_dict.items():
            if log_video:
                render_imgs.append(agent.render(info[actor_id]['reward_debug'], info[actor_id]['terminal_debug']))
            if done[actor_id] and (actor_id not in ep_stat_dict):
                ep_stat_dict[actor_id] = info[actor_id]['episode_stat']
                ep_event_dict[actor_id] = info[actor_id]['episode_event']

                # Add intersection-over-union metrics
                custom_metrics = agent.compute_metrics()
                ep_stat_dict[actor_id] = {**ep_stat_dict[actor_id], **custom_metrics}

        if len(list_render) > 15000:
            del list_render[0]
        if log_video:
            list_render.append(tile_images(render_imgs))

        timestamp = env.timestamp
        if max_step and timestamp['step'] > max_step:
            break

        counter += 1
        if counter == warm_start:
            t0 = time()

    run_fps = (counter - warm_start) / (time() - t0)
    print(f'FPS: {run_fps:.1f}')

    return list_render, ep_stat_dict, ep_event_dict, timestamp


@hydra.main(config_path='config', config_name='evaluate')
def main(cfg: DictConfig):
    log.setLevel(getattr(logging, cfg.log_level.upper()))
    if cfg.kill_running:
        server_utils.kill_carla(cfg.port)

    # start carla servers
    server_manager = server_utils.CarlaServerManager(cfg.carla_sh_path, port=cfg.port)
    server_manager.start()

    # single actor, place holder for multi actors
    agents_dict = {}
    obs_configs = {}
    reward_configs = {}
    terminal_configs = {}
    agent_names = []
    for ev_id, ev_cfg in cfg.actors.items():
        agent_names.append(ev_cfg.agent)
        cfg_agent = cfg.agent[ev_cfg.agent]
        OmegaConf.save(config=cfg_agent, f='config_agent.yaml')
        AgentClass = config_utils.load_entry_point(cfg_agent.entry_point)
        agents_dict[ev_id] = AgentClass('config_agent.yaml')
        obs_configs[ev_id] = agents_dict[ev_id].obs_configs

        # get obs_configs from agent
        reward_configs[ev_id] = OmegaConf.to_container(ev_cfg.reward)
        terminal_configs[ev_id] = OmegaConf.to_container(ev_cfg.terminal)

    # check h5 birdview maps have been generated
    config_utils.check_h5_maps(cfg.test_suites, obs_configs, cfg.carla_sh_path)

    # resume env_idx from checkpoint.txt
    last_checkpoint_path = f'{hydra.utils.get_original_cwd()}/outputs/port_{cfg.port}_checkpoint.txt'
    if cfg.resume and os.path.isfile(last_checkpoint_path):
        with open(last_checkpoint_path, 'r') as f:
            env_idx = int(f.read())
        log.info(f'Resume from env_idx {env_idx}')
    else:
        env_idx = 0

    # This is used when resuming benchmarking to indicate that all scenarios were evaluated.
    if env_idx >= len(cfg.test_suites):
        log.info(f'Finished! env_idx: {env_idx}')
        server_manager.stop()
        return

    # resume task_idx from ep_stat_buffer_{env_idx}.json
    ep_state_buffer_json = f'{hydra.utils.get_original_cwd()}/outputs/port_{cfg.port}_ep_stat_buffer_{env_idx}.json'
    if cfg.resume and os.path.isfile(ep_state_buffer_json):
        ep_stat_buffer = json.load(open(ep_state_buffer_json, 'r'))
        ckpt_task_idx = len(ep_stat_buffer['hero'])
        log.info(f'Resume from task_idx {ckpt_task_idx}')
    else:
        ckpt_task_idx = 0
        ep_stat_buffer = {}
        for actor_id in agents_dict.keys():
            ep_stat_buffer[actor_id] = []
        log.info(f'Start new env from task_idx {ckpt_task_idx}')

    # compose suite_name
    env_setup = OmegaConf.to_container(cfg.test_suites[env_idx])
    suite_name = '-'.join(agent_names) + '_' + env_setup['env_id']
    for k in sorted(env_setup['env_configs']):
        suite_name = suite_name + '_' + str(env_setup['env_configs'][k])

    log.info(f"Start Benchmarking! env_idx: {env_idx}, suite_name: {suite_name}")

    # make directories
    diags_dir = Path('diagnostics') / suite_name
    agents_log_dir = Path('agents_log') / suite_name
    video_dir = Path('videos') / suite_name
    diags_dir.mkdir(parents=True, exist_ok=True)
    agents_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # make env
    env = gym.make(env_setup['env_id'], obs_configs=obs_configs, reward_configs=reward_configs,
                   terminal_configs=terminal_configs, host=cfg.host, port=cfg.port,
                   seed=cfg.seed, no_rendering=cfg.no_rendering, **env_setup['env_configs'])

    # init wandb
    wandb.init(project=cfg.wb_project, name=suite_name, group=cfg.wb_group, tags=cfg.wb_tags)
    wandb.config.update(OmegaConf.to_container(cfg))
    wandb.save('./config_agent.yaml')

    # loop through each route
    for task_idx in range(ckpt_task_idx, env.num_tasks):
        env.set_task_idx(task_idx)
        run_name = f"{env.task['weather']}_{env.task['route_id']:02d}"

        list_render, ep_stat_dict, ep_event_dict, timestamp = run_single(
            run_name, env, agents_dict, agents_log_dir, cfg.log_video)
        # log video
        if cfg.log_video:
            video_path = (video_dir / f'{run_name}.mp4').as_posix()
            encoder = ImageEncoder(video_path, list_render[0].shape, 2*CARLA_FPS, 2*CARLA_FPS)
            for im in list_render:
                encoder.capture_frame(im)
            encoder.close()
            encoder = None
            wandb.log({f'video/{task_idx}-{run_name}': wandb.Video(video_path)}, step=task_idx)

        # dump events
        diags_json_path = (diags_dir / f'{task_idx:03}_{run_name}.json').as_posix()
        with open(diags_json_path, 'w') as fd:
            json.dump(ep_event_dict, fd, indent=4, sort_keys=False)

        # save diags and agents_log
        wandb.save(diags_json_path)

        # save time
        wandb.log({'time/total_step': timestamp['step'],
                   'time/fps':  timestamp['step'] / timestamp['relative_wall_time']
                   }, step=task_idx)

        # save statistics
        for actor_id, ep_stat in ep_stat_dict.items():
            ep_stat_buffer[actor_id].append(ep_stat)
            log_dict = {}
            for k, v in ep_stat.items():
                k_actor = f'{actor_id}/{k}'
                log_dict[k_actor] = v
            wandb.log(log_dict, step=task_idx)

        with open(ep_state_buffer_json, 'w') as fd:
            json.dump(ep_stat_buffer, fd, indent=4, sort_keys=True)
        wandb.save(ep_state_buffer_json)
        # clean up
        list_render.clear()
        ep_stat_dict = None
        ep_event_dict = None

    # close env
    env.close()
    env = None
    server_manager.stop()

    # log after suite is completed
    table_data = []
    ep_stat_keys = None
    for actor_id, list_ep_stat in json.load(open(ep_state_buffer_json, 'r')).items():
        avg_ep_stat = WandbCallback.get_avg_ep_stat(list_ep_stat)
        data = [suite_name, actor_id, str(len(list_ep_stat))]
        if ep_stat_keys is None:
            ep_stat_keys = list(avg_ep_stat.keys())
        data += [f'{avg_ep_stat[k]:.4f}' for k in ep_stat_keys]
        table_data.append(data)

    # Save all run statistics
    table_columns = ['Suite', 'actor_id', 'n_episode'] + ep_stat_keys
    wandb.log({'table/summary': wandb.Table(data=table_data, columns=table_columns)})

    with open(last_checkpoint_path, 'w') as f:
        f.write(f'{env_idx+1}')

    log.info(f"Finished Benchmarking env_idx {env_idx}, suite_name: {suite_name}")
    if env_idx+1 == len(cfg.test_suites):
        log.info(f"Finished, {env_idx+1}/{len(cfg.test_suites)}")
        return
    else:
        log.info(f"Not finished, {env_idx+1}/{len(cfg.test_suites)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
    log.info("evaluate.py DONE!")
