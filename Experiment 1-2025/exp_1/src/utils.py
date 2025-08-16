import omegaconf
from omegaconf import OmegaConf


def load_config(config_path: str = "config_roberta.yaml") -> omegaconf.DictConfig:
    configs = OmegaConf.load(config_path)
    return configs
