import os

import luigi

pwd = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(os.path.dirname(pwd), "configs")


class aframe(luigi.Config):
    """
    Global config for aframe experiments
    """

    ifos = luigi.ListParameter(default=["H1", "L1"])


class wandb(luigi.Config):
    api_key = luigi.Parameter(default=os.getenv("WANDB_API_KEY", ""))
    entity = luigi.Parameter(default=os.getenv("WANDB_ENTITY", ""))
    project = luigi.Parameter(default=os.getenv("WANDB_PROJECT", "aframe"))
    name = luigi.Parameter(default=os.getenv("WANDB_NAME", ""))
    group = luigi.Parameter(default=os.getenv("WANDB_GROUP", ""))
    tags = luigi.Parameter(default=os.getenv("WANDB_TAGS", ""))


class s3(luigi.Config):
    endpoint_url = luigi.Parameter(default=os.getenv("AWS_ENDPOINT_URL"))


class Defaults:
    TRAIN = os.path.join(config_dir, "train.yaml")