import argparse
import logging
import os

import lightning.pytorch as pl
from lightning.fabric.utilities.seed import seed_everything
from train.model import Aframe

from aframe.architectures import Zoo
from aframe.logging import configure_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101588)

    args = parser.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    configure_logging(f"{args.run_dir}/train.log")

    # TODO: add worker_init_fn in ml4gw
    logging.info(f"Setting global seed to {args.seed}")
    seed_everything(args.seed, workers=True)

    aframe = Aframe(
        arch=Zoo.ResNet(norm_groups=[3, 4, 6, 3], norm_groups=16),
        data_dir="/home/alec.gunny/aframe/data/train",
        ifos=["H1", "L1"],
        valid_frac=0.25,
        batch_size=384,
        max_lr=0.000585,
        lr_ramp_epochs=23,
        snr_thresh=4,
        max_snr=100,
        snr_alpha=3,
        kernel_length=1.5,
        psd_length=8,
        fduration=1,
        highpass=32,
        fftlength=None,
        waveform_prob=0.277,
        swap_frac=0.014,
        mute_frac=0.055,
        trigger_pad=-0.75,
        valid_stride=0.5,
        num_valid_views=4,
        valid_livetime=3600 * 8,
    )

    logger = pl.loggers.CSVLogger(args.run_dir, flush_logs_every_n_steps=10)
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=200,
        min_epochs=10,
        precision="16-mixed",
        devices=[args.gpu],
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=20,
        benchmark=True,
    )
    trainer.fit(aframe)


if __name__ == "__main__":
    main()
