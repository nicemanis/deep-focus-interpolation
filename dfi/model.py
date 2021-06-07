from keras.optimizers import Adam

from dfi.losses import Losses
from dfi.metrics import Metrics
from dfi.unet import unet


def load_model(hparams, model_path):
    model = create_model(hparams)
    model.load_weights(model_path)
    return model


def create_model(hparams):
    model, inputs = unet(
        type=hparams.model.type,
        num_blocks=hparams.model.num_blocks,
        num_filters=hparams.model.num_filters,
        convs_per_block=hparams.model.convs_per_block,
    )

    metrics = Metrics(hparams)
    losses = Losses(hparams, metrics)

    if hparams.model.type == "target":
        mse = losses.get_mse()
    else:
        mse = losses.get_residual_mse(inputs)

    model.compile(
        Adam(learning_rate=hparams.training.learning_rate),
        loss=losses.get_loss(inputs),
        metrics=[mse] + metrics.get_metrics(inputs)
    )

    model.summary()

    return model
