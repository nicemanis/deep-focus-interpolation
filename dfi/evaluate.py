import os
import math
import argparse

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from dfi.data import load_data
from dfi.model import load_model
from dfi.hparams import create_hparams


def get_results_summary(results):
    """
    :param results: a list of triplest (mse, psnr, ssim)
    :return:
    """
    total = len(results)
    five_p = math.ceil(total * 0.05)
    results.sort(key=lambda x: x[0], reverse=True)
    avg_mse = np.average([r[0] for r in results])
    avg_psnr = np.average([r[1] for r in results])
    avg_ssim = np.average([r[2] for r in results])
    w5p = results[:five_p]
    w5p_mse = np.average([r[0] for r in w5p])
    w5p_psnr = np.average([r[1] for r in w5p])
    w5p_ssim = np.average([r[2] for r in w5p])

    avg_mse = f"{round(avg_mse, 3):.3f}"
    avg_psnr = f"{round(avg_psnr, 3):.3f}"
    avg_ssim = f"{round(avg_ssim, 3):.3f}"
    w5p_mse = f"{round(w5p_mse, 3):.3f}"
    w5p_psnr = f"{round(w5p_psnr, 3):.3f}"
    w5p_ssim = f"{round(w5p_ssim, 3):.3f}"

    return [avg_mse, avg_psnr, avg_ssim, w5p_mse, w5p_psnr, w5p_ssim]


def calculate_baseline(x, y, hparams):
    noint_results = []
    lerp_results = []
    for i in tqdm(range(len(x))):
        x1 = x[i, :, :, 0]
        x2 = x[i, :, :, 1]
        y_true = y[i, :, :, 0]
        avg = (x1 + x2) / 2  # calculate average before interp

        # interpolate to 0-255 value range for correct metric calculation
        x1 = np.uint8(np.interp(x1, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))
        x2 = np.uint8(np.interp(x2, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))
        y_true = np.uint8(np.interp(y_true, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))
        avg = np.uint8(np.interp(avg, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))

        # calculate no-interpolation metrics
        if mean_squared_error(x1, y_true) < mean_squared_error(x2, y_true):
            closest_x = x1
        else:
            closest_x = x2

        mse = mean_squared_error(y_true, closest_x)
        psnr = peak_signal_noise_ratio(y_true, closest_x)
        ssim = structural_similarity(y_true, closest_x)
        noint_results.append((mse, psnr, ssim))

        # calculate linear-interpolation metrics
        mse = mean_squared_error(y_true, avg)
        psnr = peak_signal_noise_ratio(y_true, avg)
        ssim = structural_similarity(y_true, avg)
        lerp_results.append((mse, psnr, ssim))

    total = len(noint_results)
    five_p = math.ceil(total * 0.05)
    noint_results.sort(key=lambda x: x[0], reverse=True)
    avg_mse = np.average([r[0] for r in noint_results])
    avg_psnr = np.average([r[1] for r in noint_results])
    avg_ssim = np.average([r[2] for r in noint_results])
    w5p = noint_results[:five_p]
    w5p_mse = np.average([r[0] for r in w5p])
    w5p_psnr = np.average([r[1] for r in w5p])
    w5p_ssim = np.average([r[2] for r in w5p])
    no_interp_results = {
        "results": [avg_mse, avg_mse, avg_psnr, avg_ssim],
        "worst5_results": [w5p_mse, w5p_mse, w5p_psnr, w5p_ssim],
        "samples": x.shape[0]
    }

    total = len(lerp_results)
    five_p = math.ceil(total * 0.05)
    lerp_results.sort(key=lambda x: x[0], reverse=True)
    avg_mse = np.average([r[0] for r in lerp_results])
    avg_psnr = np.average([r[1] for r in lerp_results])
    avg_ssim = np.average([r[2] for r in lerp_results])
    w5p = lerp_results[:five_p]
    w5p_mse = np.average([r[0] for r in w5p])
    w5p_psnr = np.average([r[1] for r in w5p])
    w5p_ssim = np.average([r[2] for r in w5p])
    linear_interp_results = {
        "results": [avg_mse, avg_mse, avg_psnr, avg_ssim],
        "worst5_results": [w5p_mse, w5p_mse, w5p_psnr, w5p_ssim],
        "samples": x.shape[0]
    }

    return no_interp_results, linear_interp_results


def evaluate(hparams, model_path):
    model = load_model(hparams, model_path)
    model_name = model_path.split(os.sep)[-1]

    # load data
    if hparams.model.type == "residual":
        x, y, yt = load_data(hparams, subset="test")
    else:
        x, y = load_data(hparams, subset="test")

    model_results = []
    noint_results = []
    lerp_results = []

    for i in tqdm(range(len(x))):
        x1 = x[i, :, :, 0]
        x2 = x[i, :, :, 1]

        if hparams.model.type == "residual":
            r_pred = model.predict(x[i:i+1, :, :, :])
            y1_pred = x1 + r_pred[0, :, :, 0]
            y2_pred = x2 + r_pred[0, :, :, 1]
            y_pred = (y1_pred + y2_pred) / 2
            y_true = yt[i, :, :, 0]
        else:
            y_pred = model.predict(x[i:i+1, :, :, :])[0, :, :, 0]
            y_true = y[i, :, :, 0]

        avg = (x1 + x2) / 2  # calculate average before interp

        # interpolate to 0-255 value range for correct metric calculation
        x1 = np.uint8(np.interp(x1, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))
        x2 = np.uint8(np.interp(x2, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))
        y_pred = np.uint8(np.interp(y_pred, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))
        y_true = np.uint8(np.interp(y_true, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))
        avg = np.uint8(np.interp(avg, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))

        # calculate model's metrics
        mse = mean_squared_error(y_true, y_pred)
        psnr = peak_signal_noise_ratio(y_true, y_pred)
        ssim = structural_similarity(y_true, y_pred)
        model_results.append((mse, psnr, ssim))

        # calculate no-interpolation metrics
        if mean_squared_error(x1, y_true) < mean_squared_error(x2, y_true):
            closest_x = x1
        else:
            closest_x = x2

        mse = mean_squared_error(y_true, closest_x)
        psnr = peak_signal_noise_ratio(y_true, closest_x)
        ssim = structural_similarity(y_true, closest_x)
        noint_results.append((mse, psnr, ssim))

        # calculate linear-interpolation metrics
        mse = mean_squared_error(y_true, avg)
        psnr = peak_signal_noise_ratio(y_true, avg)
        ssim = structural_similarity(y_true, avg)
        lerp_results.append((mse, psnr, ssim))

        # # save imgs
        # plt.imsave(os.path.join(model_name, f"{i}_x1.png"), x1)
        # plt.imsave(os.path.join(model_name, f"{i}_x2.png"), x2)
        # plt.imsave(os.path.join(model_name, f"{i}_y_pred.png"), y_pred)
        # plt.imsave(os.path.join(model_name, f"{i}_y_true.png"), y_true)
        # plt.imsave(os.path.join(model_name, f"{i}_lin.png"), avg)
        # lines = [
        #     [model_name] + [f"{num:.3f}" for num in list(model_results[-1])],
        #     ["No interp."] + [f"{num:.3f}" for num in list(noint_results[-1])],
        #     ["Lin. interp."] + [f"{num:.3f}" for num in list(lerp_results[-1])]
        # ]
        # lines = ["\t".join(l) + "\n" for l in lines]
        # with open(os.path.join(model_name, f"{i}_results.txt"), "w", encoding="utf-8") as f:
        #     f.writelines(lines)

    #######################################################################
    table = PrettyTable()
    table.field_names = ["Method", "MSE", "PSNR", "SSIM", "W5% MSE", "W5% PSNR", "W5% SSIM"]
    table.add_row([model_name] + get_results_summary(model_results))
    table.add_row(["No interp."] + get_results_summary(noint_results))
    table.add_row(["Lin. interp."] + get_results_summary(lerp_results))
    print(table)

    # with open(f"{model_name}_full_results.txt", "w", encoding="utf-8") as f:
    #     fr = [f"{r[0]} {r[1]} {r[2]}\n" for r in model_results]
    #     f.writelines(fr)
    #
    # with open(f"{model_name}_results.txt", "w", encoding="utf-8") as f:
    #     f.write(table.get_string() + "\n")

    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hparams_path', type=str, help='Path to the yaml config file')
    parser.add_argument('-m', '--model_path', type=str, help='Path to the model weights')
    args = parser.parse_args()

    hparams = create_hparams(args.hparams_path)
    evaluate(hparams, args.model_path)
