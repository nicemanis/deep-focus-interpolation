import random
import numpy as np
import tensorflow as tf


def reset_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)
