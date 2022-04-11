from munch import Munch
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.losses import mean_squared_error

from dfi.metrics import Metrics


class Losses:
    def __init__(self, hparams: Munch, metrics: Metrics):
        self.hparams = hparams
        self.metrics = metrics

    def get_mse(self):
        def mse(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true), axis=-1)

        return mse

    def get_residual_mse(self, x):
        def mse(r_true, r_pred):
            r1_true = r_true[:, :, :, 0:1]
            r2_true = r_true[:, :, :, 1:2]

            r1_pred = r_pred[:, :, :, 0:1]
            r2_pred = r_pred[:, :, :, 1:2]

            return mean_squared_error(r1_true, r1_pred) + mean_squared_error(r2_true, r2_pred)

        return mse

    def get_psnr_loss(self):
        psnr = self.metrics.get_psnr()

        def psnr_loss(y_true, _y_pred):
            return tf.reduce_mean(psnr(y_true, y_pred))

        return psnr_loss

    def get_ssim_loss(self):
        ssim = self.metrics.get_ssim()

        def ssim_loss(y_true, y_pred):
            return K.mean((1.0 - ssim(y_true, y_pred)) / 2.0)

        return ssim_loss

    def get_perceptual_loss(self):
        vgg = VGG16(include_top=False, weights="imagenet", input_shape=(None, None, 3))
        loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer(self.hparams.training.peceptual_loss_layer).output)
        loss_model.trainable = False

        def perceptual_loss(y_true, y_pred):
            y_t = K.concatenate((y_true, y_true, y_true), axis=-1)
            y_p = K.concatenate((y_pred, y_pred, y_pred), axis=-1)
            return K.mean(K.square(loss_model(y_t) - loss_model(y_p)))

        return perceptual_loss

    def get_combined_loss(self):
        mse = self.get_mse()
        ssim_loss = self.get_ssim_loss()
        perceptual_loss = self.get_perceptual_loss()

        def combined_loss(y_true, y_pred):
            return mse(y_true, y_pred) * self.hparams.training.mse_weight +\
                   ssim_loss(y_true, y_pred) * self.hparams.training.ssim_weight + \
                   perceptual_loss(y_true, y_pred) * self.hparams.training.perceptual_weight

        return combined_loss

    def get_loss(self, x):
        if self.hparams.training.loss == "mse":
            if self.hparams.model.type == "target":
                return self.get_mse()
            else:
                return self.get_residual_mse(x)
        elif self.hparams.training.loss == "ssim":
            if self.hparams.model.type == "target":
                return self.get_ssim_loss()
        elif self.hparams.training.loss == "perceptual_loss":
            if self.hparams.model.type == "target":
                return self.get_perceptual_loss()
        elif self.hparams.training.loss == "combined_loss":
            if self.hparams.model.type == "target":
                return self.get_combined_loss()
