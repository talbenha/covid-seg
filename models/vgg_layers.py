from tensorflow import keras

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose,\
                                    concatenate, LeakyReLU, UpSampling2D

# sub-parts of the U-Net MODEL


class single_conv(keras.layers.Layer):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, out_ch, kernel_size=3):
        super(single_conv, self).__init__()
        self.sconv = keras.Sequential([
            Conv2D(out_ch, kernel_size, activation='linear', padding='same', kernel_initializer='he_normal'),
            LeakyReLU()]
        )

    def call(self, x):
        x = self.sconv(x)
        return x

class concat_and_single_conv(keras.layers.Layer):
    '''(concat => conv => BN => ReLU) * 2'''

    def __init__(self, out_ch, kernel_size=3):
        super(concat_and_single_conv, self).__init__()
        self.sconv = keras.Sequential([
            Conv2D(out_ch, kernel_size, activation='linear', padding='same', kernel_initializer='he_normal'),
            LeakyReLU()]
        )

    def call(self, x, x_old):
        x1 = concatenate([x_old, x], axis=-1)
        x2 = self.sconv(x1)
        return x2



class double_conv(keras.layers.Layer):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, out_ch, kernel_size=3):
        super(double_conv, self).__init__()
        self.conv = keras.Sequential([
            Conv2D(out_ch, kernel_size, activation='linear', padding='same', kernel_initializer='he_normal'),
            LeakyReLU(),
            Conv2D(out_ch, kernel_size, activation='linear', padding='same', kernel_initializer='he_normal'),
            LeakyReLU()])


    def call(self, x):
        x = self.conv(x)
        return x


class inconv(keras.layers.Layer):
    def __init__(self, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(out_ch)

    def call(self, x):
        x = self.conv(x)
        return x


class down(keras.layers.Layer):
    def __init__(self, out_ch):
        super(down, self).__init__()
        self.mpconv = keras.Sequential([
            MaxPooling2D(pool_size=(2, 2)),
            double_conv(out_ch)]
        )

    def call(self, x, x1=None):
        if not (x1 is None):
            x = concatenate([x, x1], axis=-1)
        x = self.mpconv(x)
        return x


class single_down(keras.layers.Layer):
    def __init__(self, out_ch):
        super(single_down, self).__init__()
        self.down_conv = nn.Sequential([
            MaxPooling2D(pool_size=(2, 2)),
            single_conv(out_ch)]
        )

    def call(self, x):
        x = self.down_conv(x)
        return x





class up(keras.layers.Layer):
    def __init__(self, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')
        else:
            self.up = Conv2DTranspose(out_ch, (3, 3), strides=(2, 2), padding='same')

        self.conv = double_conv(out_ch)

    def call(self, x1, x2, x3=None):
        x1 = self.up(x1)
        if x3 is None:
            x = concatenate([x2, x1], axis=-1)
        else:
            x = concatenate([x3, x2, x1], axis=-1)
        x = self.conv(x)
        return x



class up_drop(keras.layers.Layer):
    def __init__(self, out_ch, bilinear=False, drop_prob=0.5):
        super(up_drop, self).__init__()

        if bilinear:
            self.up = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')
        else:
            self.up = Conv2DTranspose(out_ch, (3, 3), strides=(2, 2), padding='same')
        self.drop = Dropout(drop_prob)
        self.conv = double_conv(out_ch)

    def call(self, x1, x2, x3=None):
        x1 = self.up(x1)
        if x3 is None:
            x = concatenate([x2, x1], axis=-1)
        else:
            x = concatenate([x3, x2, x1], axis=-1)

        x= self.drop(x)

        x = self.conv(x)
        return x


class outconv(keras.layers.Layer):
    def __init__(self, out_ch, kernel_size=3, sigmoid_flag=True):
        super(outconv, self).__init__()
        if sigmoid_flag:
            self.conv = Conv2D(out_ch, kernel_size, activation='sigmoid', padding='same', kernel_initializer='he_normal')
        else:
            self.conv = Conv2D(out_ch, kernel_size, activation='linear', padding='same',
                               kernel_initializer='he_normal')

    def call(self, x):
        x = self.conv(x)
        return x
