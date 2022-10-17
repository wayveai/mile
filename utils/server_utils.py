"""Adapted from https://github.com/zhejz/carla-roach CC-BY-NC 4.0 license."""

import subprocess
import os
import time
from omegaconf import OmegaConf
import logging
log = logging.getLogger(__name__)

from mile.constants import CARLA_FPS


def kill_carla(port=2005):
    # The command below kills ALL carla processes
    #kill_process = subprocess.Popen('killall -9 -r CarlaUE4-Linux', shell=True)

    # This one only kills processes linked to a certain port
    kill_process = subprocess.Popen(f'fuser -k {port}/tcp', shell=True)
    log.info(f"Killed Carla Servers on port {port}!")
    kill_process.wait()
    time.sleep(1)


class CarlaServerManager():
    def __init__(self, carla_sh_str, port=2000, configs=None, t_sleep=5):
        self._carla_sh_str = carla_sh_str
        self.port = port
        # self._root_save_dir = root_save_dir
        self._t_sleep = t_sleep
        self.env_configs = []

        if configs is None:
            cfg = {
                'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'port': port,
            }
            self.env_configs.append(cfg)
        else:
            for cfg in configs:
                for gpu in cfg['gpu']:
                    single_env_cfg = OmegaConf.to_container(cfg)
                    single_env_cfg['gpu'] = gpu
                    single_env_cfg['port'] = port
                    self.env_configs.append(single_env_cfg)
                    port += 5

    def start(self):
        kill_carla(self.port)
        for cfg in self.env_configs:
            cmd = f'CUDA_VISIBLE_DEVICES={cfg["gpu"]} bash {self._carla_sh_str} ' \
                f'-fps={CARLA_FPS} -quality-level=Epic -carla-rpc-port={cfg["port"]}'
            log.info(cmd)
            server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        time.sleep(self._t_sleep)

    def stop(self):
        kill_carla(self.port)
        time.sleep(self._t_sleep)
