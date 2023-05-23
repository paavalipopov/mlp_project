from omegaconf import OmegaConf
import hydra

from src.utils import get_resumed_cfg
from scripts.run_experiments import start

@hydra.main(version_base=None, config_path="../src/conf", config_name="resume_config")
def resume(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg = get_resumed_cfg(cfg)
    start(cfg)

if __name__ == "__main__":
    resume()