from models.Unet_vgg_Model import *
from tensorflow.keras.layers import Input, concatenate


class Wnet_vgg(keras.layers.Layer):
    UNET_BIN=0
    UNET_MULTI=1
    WNET=2

    def __init__(self,  n_classes=4 ,input_size=[None, None, 1]):
        super(Wnet_vgg, self).__init__()
        self.in1 = Input(input_size)
        self.unet_binary = Unet_binary()
        self.unet_multi = Unet_multi(n_classes=n_classes)

    def call(self, x):
        x1 = self.unet_binary(x)
        x2 = concatenate([x, x1], axis=-1)
        x3 = self.unet_multi(x2)
        return x1, x3

    def build(self):
        out1 = self.unet_binary(self.in1)
        merge_final = concatenate([self.in1, out1], axis=-1)
        out2 = self.unet_multi(merge_final)
        return tf.keras.Model(inputs=self.in1, outputs=[out1, out2])