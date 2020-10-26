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
from tensorflow.keras.regularizers import l2

class DeepGoogLeNet:

    @staticmethod
    def conv_module(x, filters, kernel_size, strides, channel_dim, padding="same", reg=0.0005):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=channel_dim)(x)
        x = Activation("relu")(x)
        return x

    @staticmethod
    def inception_module(x, filters1x1, filters3x3_reduce, filters3x3, filters5x5_reduce, filters5x5, filters1x1_proj, channel_dim, reg=0.0005):
        # First branch 1x1 convolutions
        branch_1 = DeepGoogLeNet.conv_module(x, filters1x1, (1, 1), (1, 1), channel_dim, reg=reg)

        # Second branch 1x1 convolutions => 3x3 convolutions
        branch_2 = DeepGoogLeNet.conv_module(x, filters3x3_reduce, (1, 1), (1, 1), channel_dim, reg=reg)
        branch_2 = DeepGoogLeNet.conv_module(branch_2, filters3x3, (3, 3), (1, 1), channel_dim, reg=reg)

        # Third branch 1x1 convolutions => 5x5 convolutions
        branch_3 = DeepGoogLeNet.conv_module(x, filters5x5_reduce, (1, 1), (1, 1), channel_dim, reg=reg)
        branch_3 = DeepGoogLeNet.conv_module(branch_3, filters5x5, (5, 5), (1, 1), channel_dim, reg=reg)

        # Fourth branch 3x3 max pooling => 1x1 convolution (projection)
        branch_4 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_4 = DeepGoogLeNet.conv_module(branch_4, filters1x1_proj, (1, 1), (1, 1), channel_dim, reg=reg)

        # Concatentate results
        x = concatenate([branch_1, branch_2, branch_3, branch_4], axis=channel_dim)
        return x

    @staticmethod
    def build(height, width, channels, classes, reg=0.0005):
        input_shape = (height, width, channels)
        channel_dim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (channels, height, width)
            channel_dim = 1

        inputs = Input(shape=input_shape)
        x = DeepGoogLeNet.conv_module(inputs, 64, (5, 5), (1, 1), channel_dim, reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = DeepGoogLeNet.conv_module(x, 64, (1, 1), (1, 1), channel_dim, reg=reg)
        x = DeepGoogLeNet.conv_module(x, 192, (3, 3), (1, 1), channel_dim, reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

        x = DeepGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, channel_dim, reg=reg)
        x = DeepGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, channel_dim, reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

        x = DeepGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, channel_dim, reg=reg)
        x = DeepGoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, channel_dim, reg=reg)
        x = DeepGoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, channel_dim, reg=reg)
        x = DeepGoogLeNet.inception_module(x, 112, 144, 288, 32, 64, 64, channel_dim, reg=reg)
        x = DeepGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, channel_dim, reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

        x = AveragePooling2D((4, 4))(x)
        x = Dropout(0.4)(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg), activation="softmax")(x)

        return Model(inputs, x, name="deepgooglenet")
