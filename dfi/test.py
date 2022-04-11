import math
from dfi.model import load_model
from dfi.data import load_data
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


def test(model=None, hparams=None):
    if model is None:
        from dfi.hparams import create_hparams
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-hp', '--hparams_path', type=str, help='Path to the yaml config file')
        parser.add_argument('-m', '--model_path', type=str, help='Path to the model weights')
        args = parser.parse_args()

        hparams = create_hparams(args.hparams_path)
        model = load_model(hparams, args.model_path)

    # load data
    if hparams.model.type == "residual":
        x, y, yt = load_data(hparams, subset="test")
    else:
        x, y = load_data(hparams, subset="test")

    model_results = []

    for i in tqdm(range(len(x))):
        x1 = x[i, :, :, 0]
        x2 = x[i, :, :, 1]

        if hparams.model.type == "residual":
            r_pred = model.predict(x[i:i + 1, :, :, :])
            y1_pred = x1 + r_pred[0, :, :, 0]
            y2_pred = x2 + r_pred[0, :, :, 1]
            y_pred = (y1_pred + y2_pred) / 2
            y_true = yt[i, :, :, 0]
        else:
            y_pred = model.predict(x[i:i + 1, :, :, :])[0, :, :, 0]
            y_true = y[i, :, :, 0]

        # interpolate to 0-255 value range for correct metric calculation
        y_pred = np.uint8(np.interp(y_pred, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))
        y_true = np.uint8(np.interp(y_true, [hparams.data.norm_min, hparams.data.norm_max], [0, 255]))

        # calculate model's metrics
        mse = mean_squared_error(y_true, y_pred)
        psnr = peak_signal_noise_ratio(y_true, y_pred)
        ssim = structural_similarity(y_true, y_pred)
        model_results.append((mse, psnr, ssim))

    total = len(model_results)
    five_p = math.ceil(total * 0.05)
    model_results.sort(key=lambda x: x[0], reverse=True)
    avg_mse = np.average([r[0] for r in model_results])
    avg_psnr = np.average([r[1] for r in model_results])
    avg_ssim = np.average([r[2] for r in model_results])
    w5p = model_results[:five_p]
    w5p_mse = np.average([r[0] for r in w5p])
    w5p_psnr = np.average([r[1] for r in w5p])
    w5p_ssim = np.average([r[2] for r in w5p])

    results_dict = {
        "results": [avg_mse, avg_mse, avg_psnr, avg_ssim],
        "worst5_results": [w5p_mse, w5p_mse, w5p_psnr, w5p_ssim],
        "samples": x.shape[0]
    }

    return results_dict
