from keras.models import *
from keras.layers import *


def unet(type="target", num_blocks=1, num_filters=None, convs_per_block=1):
    """
    Creates a U-Net model for focus interpolation.
    :param str type: model type - "target" or "residual", determines the output layer
    :param int num_blocks: number of down-sampling blocks
    :param int convs_per_block: number of convolutional layers per block
    :param list num_filters: a list of number of filters for each down-sampling block, default: [16]. The center block
    has twice as many filters as the last down-sampling block. Up-sampling blocks have the same amount of filters as the
    down-sampling blocks.
    :return:
    """
    if num_filters is None:
        num_filters = [2 ** (4 + i) for i in range(num_blocks)]
    if not isinstance(num_filters, list):
        raise TypeError(f"Argument filters must be of type list not {type(num_filters)}")
    if num_blocks != len(num_filters):
        raise ValueError("If argument filters is not None, then num_blocks should be equal to len(filters)")
    inputs = Input(shape=(None, None, 2), name="input")
    x = inputs
    down_outputs = []
    # Down-sampling blocks
    for i in range(num_blocks):
        # Convolutional operation
        for j in range(convs_per_block):
            x = Conv2D(num_filters[i], 3, padding="same", kernel_initializer="he_normal",
                       name=f"{num_filters[i]}_down_conv{j + 1}")(x)
        x = BatchNormalization(name=f"{num_filters[i]}_down_bn")(x)
        down = Activation("relu", name=f"{num_filters[i]}_down_relu")(x)
        down_outputs.append(down)
        # Down-sampling operation
        x = Dropout(0.5, name=f"{num_filters[i]}_down_dropout")(down)
        x = MaxPooling2D(pool_size=(2, 2), name=f"{num_filters[i]}_down_pool")(x)

    # Center
    center_filters = num_filters[-1] * 2
    for j in range(convs_per_block):
        x = Conv2D(center_filters, 3, padding="same", kernel_initializer="he_normal",
                   name=f"{center_filters}_center_conv{j + 1}")(x)
    x = BatchNormalization(name=f"{center_filters}_center_bn")(x)
    x = Activation("relu", name=f"{center_filters}_center_relu")(x)

    # Up-sampling blocks
    for i in range(num_blocks):
        # Up-sampling operation
        down = down_outputs[-(i + 1)]
        x = Dropout(0.5, name=f"{num_filters[-(i + 1)]}_up_dropout")(x)
        up = UpSampling2D(size=(2, 2), name=f"{num_filters[-(i + 1)]}_up_sampling")(x)
        # Concatenate operation
        x = concatenate([down, up], axis=3, name=f"{num_filters[-(i + 1)]}_concat")
        # Convolutional operation
        for j in range(convs_per_block):
            x = Conv2D(num_filters[-(i + 1)], 3, padding="same", kernel_initializer="he_normal",
                       name=f"{num_filters[-(i + 1)]}_up_conv{j + 1}")(x)
        x = BatchNormalization(name=f"{num_filters[-(i + 1)]}_up_bn")(x)
        x = Activation("relu", name=f"{num_filters[-(i + 1)]}_up_relu")(x)

    # Output
    if type == "target":
        x = Conv2D(1, 3, padding="same", kernel_initializer="he_normal", name="output_conv")(x)
        x = BatchNormalization(name="output_bn")(x)
        outputs = Activation("sigmoid", name="output_activation")(x)
    else:
        x = Conv2D(2, 3, padding="same", kernel_initializer="he_normal", name="output_conv")(x)
        x = BatchNormalization(name="output_bn")(x)
        outputs = Activation("tanh", name="output_activation")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, inputs
