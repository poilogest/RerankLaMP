import os
from omegaconf import OmegaConf

def _get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))

def load_config(base_path = "configs/base.yaml", override_paths = []):
    project_root = _get_project_root()
    base_path = os.path.join(project_root, base_path)
    override_paths = [os.path.join(project_root, path) for path in override_paths]

    config = OmegaConf.load(base_path)
    for path in override_paths:
        config = OmegaConf.merge(config, OmegaConf.load(path))
    return config

if __name__ == "__main__":
    print("Project root:", _get_project_root())
    config = load_config()
    print(OmegaConf.to_yaml(config))
    