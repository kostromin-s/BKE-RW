import os
import hydra
import torch


@hydra.main(version_base=None, config_path="config", config_name="kg_build")
def main(cfg):
    print("Loading data...")