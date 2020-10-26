from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model


class MiniGoogLeNet:

    @staticmethod
    def conv_module(x, filters, kernel_size, strides, channel_dim, padding="same"):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=channel_dim)(x)
        x = Activation("relu")(x)
        return x

    @staticmethod
    def inception_module(x, filters1x1, filters3x3, channel_dim):
        conv_1x1 = MiniGoogLeNet.conv_module(x, filters1x1, (1, 1), (1, 1), channel_dim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, filters3x3, (3, 3), (1, 1), channel_dim)
        x = concatenate([conv_1x1, conv_3x3], axis=channel_dim)
        return x

    @staticmethod
    def downsample_module(x, filters, channel_dim):
        conv_3x3 = MiniGoogLeNet.conv_module(x, filters, (3, 3), (2, 2), channel_dim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=channel_dim)
        return x

    @staticmethod
    def build(height, width, channels, classes):
        input_shape = (height, width, channels)
        channel_dim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (channels, height, width)
            channel_dim = 1

        inputs = Input(shape=input_shape)
        x = MiniGoogLeNet.conv_module(inputs, 96, (3, 3), (1, 1), channel_dim)
        x = MiniGoogLeNet.inception_module(x, 32, 32, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 32, 48, channel_dim)
        x = MiniGoogLeNet.downsample_module(x, 80, channel_dim)

        x = MiniGoogLeNet.inception_module(x, 112, 48, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, channel_dim)
        x = MiniGoogLeNet.downsample_module(x, 96, channel_dim)

        x = MiniGoogLeNet.inception_module(x, 176, 160, channel_dim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, channel_dim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(classes, activation="softmax")(x)

        return Model(inputs, x, name="minigooglenet")
