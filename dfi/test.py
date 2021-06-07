import numpy as np

from dfi.data import load_data


def test(model, hparams):
    x, y = load_data(hparams, subset="test")

    test_results = model.evaluate(x, y, batch_size=hparams.training.batch_size)

    num_samples = x.shape[0]
    five_percent_samples = int(round(num_samples * 0.05))
    worst5_results = np.zeros(4)
    results = []
    for i in range(num_samples):
        result = model.evaluate(x[i:i+1, :, :, :], y[i:i+1, :, :, :], verbose=0)
        results.append(result)

    results = sorted(results, key=lambda k: k[0], reverse=True)
    for i in range(five_percent_samples):
        worst5_results += results[i]

    worst5_results /= five_percent_samples

    print(test_results)
    print(worst5_results)

    results_dict = {
        "results": test_results,
        "worst5_results": worst5_results,
        "samples": x.shape[0]
    }

    return results_dict
