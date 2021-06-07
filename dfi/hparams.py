import yaml
import munch


def create_hparams(path):
    with open(path) as f:
        yaml_params = yaml.safe_load(f)

    return munch.munchify(yaml_params)
