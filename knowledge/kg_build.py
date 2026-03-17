import os
import hydra
from omegaconf import DictConfig

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ROOT_DIR = get_project_root()

@hydra.main(version_base=None, config_path=os.path.join(ROOT_DIR, "config"), config_name="kg_build")
def build_kg(cfg: DictConfig):
    print("Building the knowledge graph with the following configuration:")
    print(cfg)

if __name__ == "__main__":
    build_kg()