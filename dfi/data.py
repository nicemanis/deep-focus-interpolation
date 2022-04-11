import numpy as np


def load_raw_input(hparams, z=16, subset="train", idxs=None):
    # replace with a different dataset if necessary
    # if hparams.data.dataset == "bbbc006":
    from dfi.bbbc006_v1 import load_imgs

    return load_imgs(z, subset, idxs=idxs)


def load_z_level(hparams, z_lvl, subset="train", idxs=None):
    # replace with a different dataset if necessary
    # if hparams.data.dataset == "bbbc006":
    from dfi.bbbc006_v1 import load_imgs

    return np.float32(
        np.expand_dims(np.array(load_imgs(hparams, z_lvl, subset, idxs)), axis=-1)
    )


def load_data(hparams, subset="train", idxs=None):
    # replace with a different dataset if necessary
    # if hparams.data.dataset == "bbbc006":
    from dfi.bbbc006_v1 import load_imgs

    x = None
    y = None
    for triplet in hparams.data.z_triplets:
        z_x1, z_y, z_x2 = triplet

        if x is None:
            x = np.float32(
                np.stack((np.array(load_imgs(hparams, z_x1, subset, idxs)),
                          np.array(load_imgs(hparams, z_x2, subset, idxs))), axis=-1)
            )
            y = np.float32(
                np.expand_dims(np.array(load_imgs(hparams, z_y, subset, idxs)), axis=-1)
            )
        else:
            x = np.concatenate((x, np.float32(
                np.stack((np.array(load_imgs(hparams, z_x1, subset, idxs)),
                          np.array(load_imgs(hparams, z_x2, subset, idxs))), axis=-1))))
            y = np.concatenate((y, np.float32(
                np.expand_dims(np.array(load_imgs(hparams, z_y, subset, idxs)), axis=-1))))

    if hparams.model.type == "residual":
        yr = np.stack((y[:, :, :, 0] - x[:, :, :, 0], y[:, :, :, 0] - x[:, :, :, 1]), axis=-1)
        return x, yr, y
    else:
        return x, y
