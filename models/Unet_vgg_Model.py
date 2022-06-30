from models.vgg_layers import *




class Unet_binary(keras.layers.Layer):
    def __init__(self,  n_classes=1, n_channels=64):
        super(Unet_binary, self).__init__()
        self.inc = single_conv(3, kernel_size=1)

        vgg_model = tf.keras.applications.VGG16(
            include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
            pooling=None, classes=1000)

        for i, layer in enumerate(vgg_model.layers):
            layer._name = 'layer' + '0_' + str(i)

        self.down1_conv1 = vgg_model.layers[1]
        self.down1_conv2 = vgg_model.layers[2]

        self.pool = MaxPooling2D(pool_size=(2, 2))

        self.down2_conv1 = vgg_model.layers[4]
        self.down2_conv2 = vgg_model.layers[5]

        self.down3_conv1 = vgg_model.layers[7]
        self.down3_conv2 = vgg_model.layers[8]
        self.down3_conv3 = vgg_model.layers[9]

        self.down4_conv1 = vgg_model.layers[11]
        self.down4_conv2 = vgg_model.layers[12]
        self.down4_conv3 = vgg_model.layers[13]

        self.down5_conv1 = vgg_model.layers[15]
        self.down5_conv2 = vgg_model.layers[16]
        self.down5_conv3 = vgg_model.layers[17]

        self.up1 = up(n_channels * 8)
        self.up2 = up(n_channels * 4)
        self.up3 = up(n_channels * 2)
        self.up4 = up(n_channels)
        self.outc = outconv(n_classes, sigmoid_flag=True)

    def call(self, x0):
        x1 = self.inc(x0)

        x2 = self.down1_conv1(x1)
        x2 = self.down1_conv2(x2)
        x3 = self.pool(x2)

        x4 = self.down2_conv1(x3)
        x4 = self.down2_conv2(x4)
        x5 = self.pool(x4)

        x6 = self.down3_conv1(x5)
        x6 = self.down3_conv2(x6)
        x6 = self.down3_conv3(x6)
        x7 = self.pool(x6)

        x8 = self.down4_conv1(x7)
        x8 = self.down4_conv2(x8)
        x8 = self.down4_conv3(x8)
        x9 = self.pool(x8)

        x9 = self.down5_conv1(x9)
        x9 = self.down5_conv2(x9)
        x9 = self.down5_conv3(x9)

        x10 = self.up1(x9, x8)
        x11 = self.up2(x10, x6)
        x12 = self.up3(x11, x4)
        x13 = self.up4(x12, x2)
        x14 = self.outc(x13)
        return x14



class Unet_multi(keras.layers.Layer):
    def __init__(self,  n_classes=4, n_channels=64):
        super(Unet_multi, self).__init__()
        self.inc = single_conv(3, kernel_size=1)

        vgg_model = tf.keras.applications.VGG16(
            include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
            pooling=None, classes=1000)

        for i, layer in enumerate(vgg_model.layers):
            layer._name = 'layer' + '0_' + str(i)

        self.down1_conv1 = vgg_model.layers[1]
        self.down1_conv2 = vgg_model.layers[2]

        self.pool = MaxPooling2D(pool_size=(2, 2))

        self.down2_conv1 = vgg_model.layers[4]
        self.down2_conv2 = vgg_model.layers[5]

        self.down3_conv1 = vgg_model.layers[7]
        self.down3_conv2 = vgg_model.layers[8]
        self.down3_conv3 = vgg_model.layers[9]

        self.down4_conv1 = vgg_model.layers[11]
        self.down4_conv2 = vgg_model.layers[12]
        self.down4_conv3 = vgg_model.layers[13]

        self.down5_conv1 = vgg_model.layers[15]
        self.down5_conv2 = vgg_model.layers[16]
        self.down5_conv3 = vgg_model.layers[17]
        self.up1 = up(n_channels * 8)
        self.up2 = up(n_channels * 4)
        self.up3 = up(n_channels * 2)
        self.up4 = up(n_channels)
        self.outc = outconv(n_classes, sigmoid_flag=False)

    def call(self, x0):
        x1 = self.inc(x0)

        x2 = self.down1_conv1(x1)
        x2 = self.down1_conv2(x2)
        x3 = self.pool(x2)

        x4 = self.down2_conv1(x3)
        x4 = self.down2_conv2(x4)
        x5 = self.pool(x4)

        x6 = self.down3_conv1(x5)
        x6 = self.down3_conv2(x6)
        x6 = self.down3_conv3(x6)
        x7 = self.pool(x6)

        x8 = self.down4_conv1(x7)
        x8 = self.down4_conv2(x8)
        x8 = self.down4_conv3(x8)
        x9 = self.pool(x8)

        x9 = self.down5_conv1(x9)
        x9 = self.down5_conv2(x9)
        x9 = self.down5_conv3(x9)

        x10 = self.up1(x9, x8)
        x11 = self.up2(x10, x6)
        x12 = self.up3(x11, x4)
        x13 = self.up4(x12, x2)
        x14 = self.outc(x13)
        return x14
