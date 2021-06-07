import glob
from skimage.io import imread
import os
import pickle
import numpy as np


# TODO: specify the correct dataset location
DATA_PATH = "/foo/bar/BBBC006_v1"
# Min-max values encountered in the BBBC006_v1 dataset, used for min-max normalization
MIN_VALUE = 0
MAX_VALUE = 4096


def _get_sample_name(file_path):
    file = file_path.split("/")[-1]
    parts = file.split("_")
    return "_".join([parts[1], parts[2], parts[3][0:2]])


def _sort_data():
    with open(os.path.join(DATA_PATH, ".train_names"), "rb") as f:
        train_names = pickle.load(f)

    with open(os.path.join(DATA_PATH, ".test_names"), "rb") as f:
        test_names = pickle.load(f)

    dirs = glob.glob(os.path.join(DATA_PATH, "unsorted", "*"))
    for dirr in dirs:
        dir_name = dirr.split("/")[-1]
        print("Processing %s" % dir_name)

        files = glob.glob(os.path.join(DATA_PATH, "unsorted", dirr, "*"))
        for file in files:
            file_name = file.split("/")[-1]
            sample_name = _get_sample_name(file)

            if sample_name in train_names:
                os.rename(file, os.path.join(DATA_PATH, "train", dir_name, file_name))
            elif sample_name in test_names:
                os.rename(file, os.path.join(DATA_PATH, "test", dir_name, file_name))
            else:
                print("Unknown sample")
                exit(1)


def _get_subfolder(z=16):
    return "BBBC006_v1_images_z_{:02d}".format(int(z))


def _load_img(hparams, path):
    img = imread(path, plugin="tifffile")[0:512, 0:512]  # crop the img to square aspect ratio
    img = np.float32(np.interp(img, [MIN_VALUE, MAX_VALUE], [hparams.data.norm_min, hparams.data.norm_max]))
    return img


def load_imgs(hparams, z=16, subset="train", idxs=None):
    all_paths = glob.glob(os.path.join(DATA_PATH, subset, _get_subfolder(z), "*"))
    all_paths.sort()
    imgs = []

    if idxs is not None:
        for i in idxs:
            try:
                path = all_paths[i]
                img = _load_img(hparams, path)
                imgs.append(img)
            except IndexError:
                pass

        return imgs

    for path in all_paths:
        img = _load_img(hparams, path)
        imgs.append(img)

    return imgs
