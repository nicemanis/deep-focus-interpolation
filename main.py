import argparse

import numpy as np

from dfi.train import train
from dfi.model import create_model
from dfi.hparams import create_hparams


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hparams_path', type=str, help='Path to the yaml config file')
    args = parser.parse_args()

    try_count = 1
    hparams = create_hparams(args.hparams_path)

    hparams.training.seed = np.random.randint(1, 2 ** 32 - 1)
    model = create_model(hparams)
    train(model, hparams, session_name=f"training_session_1")
