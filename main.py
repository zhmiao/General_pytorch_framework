import os
import yaml
import numpy as np

import torch

import fire
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger


def main(config='./configs/beeants_plain_061521.yaml',
         gpus=[0], 
         session=0,
         np_threads=8,
         evaluate=None,
         seed=0):

    #############################
    # Set environment variables #
    #############################
    # Set numpy threads
    os.environ["OMP_NUM_THREADS"] = str(np_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(np_threads)
    os.environ["MKL_NUM_THREADS"] = str(np_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(np_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(np_threads)
    
    ###############################
    # Load configurations to args #
    ###############################
    args = {}
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config.items():
        setattr(args, k, v)

    pl.seed_everything(seed)

    alg = get_algorithm(args.algorithm, args)

    model = AudioNTT2020(n_mels=64, d=2048)

    learner = BYOLALearner(
        model=model, 
        lr=lr, 
        batch_size=batch_size, 
        noise=noise, 
        pre=pre, 
        shape=(64, 200),
        hidden_layer=-1,
        projection_size=512,
        projection_hidden_size=4096,
        moving_average_decay=0.99,
    )

    logger = WandbLogger(
        project="byol_{}".format(ann_type),
        save_dir="logs/",
        name="seive_{}_{}_n{}_{}".format(ann_type, pre, noise, session),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_acc_knn", mode="max", dirpath="models/{}".format(ann_type), save_top_k=1,
        filename='seive-encoder-{}-{}-n{}'.format(pre, session, noise) + '-{epoch:02d}-{valid_acc_knn:.2f}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=epochs,
        check_val_every_n_epoch=1, 
        log_every_n_steps = 50, 
        gpus=[g for g in gpus],
        logger=None if evaluate is not None else logger,
        callbacks=[lr_monitor, checkpoint_callback],
        accelerator='dp',
        # profiler="simple"
    )

    if evaluate is not None:
        learner = learner.load_from_checkpoint(checkpoint_path=evaluate)
        trainer.test(learner)
        # trainer.validate(learner)
    else:
        trainer.fit(learner)

if __name__ == '__main__':
    fire.Fire(main)