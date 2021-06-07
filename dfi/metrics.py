import tensorflow as tf
import keras.backend as K


class Metrics:
    def __init__(self, hparams):
        self.hparams = hparams

    def get_max_val(self):
        return self.hparams.data.norm_max - self.hparams.data.norm_min

    def get_y(self, x, r):
        y1 = r[:, :, :, 0:1] + x[:, :, :, 0:1]
        y2 = r[:, :, :, 1:2] + x[:, :, :, 1:2]
        y_pred = (y1 + y2) / 2

        return K.clip(y_pred, self.hparams.data.norm_min, self.hparams.data.norm_max)

    def get_psnr(self):
        def psnr(y_true, y_pred):
            # clip values to ensure correct PSNR calculation
            return tf.image.psnr(
                y_true, K.clip(y_pred, self.hparams.data.norm_min, self.hparams.data.norm_max), self.get_max_val())

        return psnr

    def get_residual_psnr(self, x):
        def psnr(r_true, r_pred):
            y_true = self.get_y(x, r_true)
            y_pred = self.get_y(x, r_pred)
            return tf.image.psnr(y_true, y_pred, self.get_max_val())

        return psnr

    def get_ssim(self):
        def ssim(y_true, y_pred):
            return tf.image.ssim(
                y_true, K.clip(y_pred, self.hparams.data.norm_min, self.hparams.data.norm_max), self.get_max_val())

        return ssim

    def get_residual_ssim(self, x):
        def ssim(r_true, r_pred):
            y_true = self.get_y(x, r_true)
            y_pred = self.get_y(x, r_pred)
            return tf.image.ssim(y_true, y_pred, self.get_max_val())

        return ssim

    def get_metrics(self, x):
        if self.hparams.model.type == "target":
            return [self.get_psnr(), self.get_ssim()]
        else:
            return [self.get_residual_psnr(x), self.get_residual_ssim(x)]
