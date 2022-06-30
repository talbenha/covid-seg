import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from backend.loss_metric_utils import nan_mask, identity
from losses.loss_functions import class_weighted_pixelwise_crossentropy_final\
    , dice_loss
from functools import partial

class SegLoss_Multi(tf.keras.losses.Loss):

    def __init__(self, config, model, name='segmentation_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = [partial(class_weighted_pixelwise_crossentropy_final,ce_weights=config["Weights_Cross_Entropy_Loss"]) ,
                        partial(dice_loss,dice_weights=config["Weights_Dice_Loss"])]
        self.config=config
        self.enable_reg = config["Enable_reg"]
        self.weight_decay = config["weight_decay"]
        self.reg = tf.keras.regularizers.l2(l=self.weight_decay)
        self.model = model
        self.weight_fn = nan_mask
        self.target_fn = identity
        self.pred_fn = identity


    def call(self, targets, prediction):

        tar = self.target_fn(targets)
        pred = self.pred_fn(prediction)
        loss = tf.math.add(tf.math.multiply(self.loss_fn[0](tar, pred) , self.config["Weights_Coeff"]) ,
                           tf.math.multiply(self.loss_fn[1](tar, pred) , (1 - self.config["Weights_Coeff"])))

        if self.enable_reg is True:
            loss = tf.math.add(loss ,self.regularization())
        return loss

    def regularization(self):
        reg_loss = tf.constant([0.0])
        for v in self.model.trainable_variables:
            reg_loss = tf.math.add(reg_loss, self.reg(v))
        return reg_loss

class SegLoss_Binary(tf.keras.losses.Loss):

    def __init__(self, config, model, name='segmentation_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = BinaryCrossentropy()
        self.enable_reg = config["Enable_reg"]
        self.weight_decay = config["weight_decay"]
        self.reg = tf.keras.regularizers.l2(l=self.weight_decay)
        self.model = model
        self.weight_fn = nan_mask
        self.target_fn = identity
        self.pred_fn = identity


    def call(self, targets, prediction):

        tar = self.target_fn(targets)
        pred = self.pred_fn(prediction)
        loss = self.loss_fn(tar, pred)
        if self.enable_reg is True:
            loss = loss + self.regularization()
        return loss

    def regularization(self):
        reg_loss = tf.constant([0.0])
        for v in self.model.trainable_variables:
            reg_loss = tf.math.add(reg_loss, self.reg(v))
        return reg_loss
