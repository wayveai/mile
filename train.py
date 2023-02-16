import git
import os
import socket
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from mile.config import get_parser, get_cfg
from mile.data.dataset import DataModule
from mile.trainer import WorldModelTrainer


class SaveGitDiffHashCallback(pl.Callback):
    def setup(self, trainer, pl_module, stage):
        repo = git.Repo()
        trainer.git_hash = repo.head.object.hexsha
        trainer.git_diff = repo.git.diff(repo.head.commit.tree)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['world_size'] = trainer.world_size
        checkpoint['git_hash'] = trainer.git_hash
        checkpoint['git_diff'] = trainer.git_diff


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    data = DataModule(cfg)
    model = WorldModelTrainer(cfg.convert_to_dict())

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)

    callbacks = [
        pl.callbacks.ModelSummary(-1),
        SaveGitDiffHashCallback(),
        pl.callbacks.LearningRateMonitor(),
        ModelCheckpoint(
            save_dir, every_n_train_steps=cfg.VAL_CHECK_INTERVAL,
        ),
    ]

    if cfg.LIMIT_VAL_BATCHES in [0, 1]:
        limit_val_batches = float(cfg.LIMIT_VAL_BATCHES)
    else:
        limit_val_batches = cfg.LIMIT_VAL_BATCHES

    replace_sampler_ddp = not cfg.SAMPLER.ENABLED

    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        accelerator='gpu',
        strategy='ddp',
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        max_epochs=None,
        max_steps=cfg.STEPS,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        val_check_interval=cfg.VAL_CHECK_INTERVAL,
        limit_val_batches=limit_val_batches,
        replace_sampler_ddp=replace_sampler_ddp,
        accumulate_grad_batches=cfg.OPTIMIZER.ACCUMULATE_GRAD_BATCHES,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    main()
