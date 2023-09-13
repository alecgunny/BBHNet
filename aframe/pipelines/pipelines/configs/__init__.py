import os

import luigi


class aframe(luigi.Config):
    """
    Global config for
    """

    ifos = luigi.ListParameter(default=["H1", "L1"])


class wandb(luigi.Config):
    api_key = luigi.Parameter(default=os.getenv("WANDB_API_KEY", ""))
    entity = luigi.Parameter(default=os.getenv("WANDB_ENTITY", ""))
    project = luigi.Parameter(default=os.getenv("WANDB_PROJECT", "aframe"))
    name = luigi.Parameter(default=os.getenv("WANDB_NAME", ""))
    group = luigi.Parameter(default=os.getenv("WANDB_GROUP", ""))
    tags = luigi.Parameter(default=os.getenv("WANDB_TAGS", ""))


pwd = os.path.dirname(os.path.abspath(__file__))


class Defaults:
    TRAIN = os.path.join(pwd, "train.yaml")
