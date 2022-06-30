import json
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from losses.losses import SegLoss_Binary, SegLoss_Multi
from models.wnet_vgg import Wnet_vgg

logger = logging.getLogger("logger")

def build_trainer(model, data, config):
    if config['trainer_name'] == "segmentation":
        trainer = SegTrainer(model, data, config)
    else:
        raise ValueError("'{}' is an invalid model name")

    return trainer


class SegTrainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        self.state = config["initial_state"]
        self.loss_fn = [SegLoss_Binary(config,self.model), SegLoss_Multi(config,self.model)]
        self.optimizers=[None, None, None]
        self.current_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.compute_grads_fun=[self.compute_grads_binary,self.compute_grads_multi,self.compute_grads_all]
        self.apply_grads_fun=[self.apply_grads0,self.apply_grads1,self.apply_grads2]

        self.train_loss=[np.zeros((config['num_epochs'][i],2)) for i in range(len(config['num_epochs']))]
        self.valid_loss=[np.zeros((config['num_epochs'][i],2)) for i in range(len(config['num_epochs']))]


        self.accumulator_loss=[tf.keras.metrics.Mean(name='Loss1'), tf.keras.metrics.Mean(name='Loss2')]
        self.accumulator_ValidLoss=[tf.keras.metrics.Mean(name='Loss1'), tf.keras.metrics.Mean(name='Loss2')]
        self.checkpoint_final = tf.train.Checkpoint(optimizer=self.current_optimizer, generator=self.model)

        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint_final, directory=f"{self.config['results_dir']}{self.config['exp_name']}/Checkpoints/", max_to_keep=3)
        if self.config['Starting_ckpt'] is not None:
            checkpoint_name = f"{self.config['results_dir']}{self.config['exp_name']}/Checkpoints/ckpt-{self.config['Starting_ckpt']}" if self.config['Starting_ckpt']>=0 \
            else self.ckpt_manager.latest_checkpoint
            self.checkpoint_final.restore(checkpoint_name)
            if(checkpoint_name):
                print(f'Restored checkpoint: {checkpoint_name.split("/")[-1]}')

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.global_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)


    def compute_grads(self, samples, targets, state=0):
        self.compute_grads_fun[state](samples,targets)


    @tf.function
    def compute_grads_binary(self, samples, targets):

        with tf.GradientTape() as tape1:
            predictions = self.model(samples, training=True)

            ''' generate the targets and apply the corresponding loss function '''

            loss = [self.loss_fn[0](targets[...,0,tf.newaxis], predictions[0]), self.loss_fn[1](targets[...,1,tf.newaxis], predictions[1])]
        grad_norm=[None,None]
        gradients = [tape1.gradient(loss[0], self.model.trainable_weights),
                     None]

        if self.config['clip_grad_norm'] is not None:
            gradients[0], grad_norm[0] = tf.clip_by_global_norm(gradients[0], self.config['clip_grad_norm'])
            with self.config['train_writer'].as_default():
                tf.summary.scalar("grad_norm1", grad_norm[0], self.global_step)
                self.global_step.assign_add(1)

        self.accumulator_loss[0](loss[0])
        self.accumulator_loss[1](loss[1])
        return gradients, predictions

    @tf.function
    def compute_grads_multi(self, samples, targets):

        with tf.GradientTape() as tape2:
            predictions = self.model(samples, training=True)

            ''' generate the targets and apply the corresponding loss function '''

            loss = [self.loss_fn[0](targets[..., 0, tf.newaxis], predictions[0]),
                    self.loss_fn[1](targets[..., 1, tf.newaxis], predictions[1])]
        grad_norm = [None, None]
        gradients = [None,
                     tape2.gradient(loss[1], self.model.trainable_weights)]

        if self.config['clip_grad_norm'] is not None:
            gradients[1], grad_norm[1] = tf.clip_by_global_norm(gradients[1], self.config['clip_grad_norm'])
            with self.config['train_writer'].as_default():

                tf.summary.scalar("grad_norm2", grad_norm[1], self.global_step)
                self.global_step.assign_add(1)

        self.accumulator_loss[0](loss[0])
        self.accumulator_loss[1](loss[1])
        return gradients, predictions

    @tf.function
    def compute_grads_all(self, samples, targets):

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            predictions = self.model(samples, training=True)

            ''' generate the targets and apply the corresponding loss function '''

            loss = [self.loss_fn[0](targets[..., 0, tf.newaxis], predictions[0]),
                    self.loss_fn[1](targets[..., 1, tf.newaxis], predictions[1])]
        grad_norm = [None, None]
        gradients = [tape1.gradient(loss[0], self.model.trainable_weights),
                     tape2.gradient(loss[1], self.model.trainable_weights)]

        if self.config['clip_grad_norm'] is not None:
            gradients[0], grad_norm[0] = tf.clip_by_global_norm(gradients[0], self.config['clip_grad_norm'])
            gradients[1], grad_norm[1] = tf.clip_by_global_norm(gradients[1], self.config['clip_grad_norm'])
            with self.config['train_writer'].as_default():
                tf.summary.scalar("grad_norm1", grad_norm[0], self.global_step)
                tf.summary.scalar("grad_norm2", grad_norm[1], self.global_step)
                self.global_step.assign_add(1)

        self.accumulator_loss[0](loss[0])
        self.accumulator_loss[1](loss[1])
        return gradients, predictions

    @tf.function
    def apply_grads0(self, gradients):
        self.optimizers[0].apply_gradients(zip(gradients[0], self.model.trainable_weights))


    @tf.function
    def apply_grads1(self, gradients):
        self.optimizers[1].apply_gradients(zip(gradients[1], self.model.trainable_weights))

    @tf.function
    def apply_grads2(self, gradients):
        self.optimizers[2].apply_gradients(zip(gradients[0], self.model.trainable_weights))
        self.optimizers[2].apply_gradients(zip(gradients[1], self.model.trainable_weights))

    def train_step(self, samples, targets, state=0):
        gradients, predictions = self.compute_grads_fun[state](samples,targets)
        self.apply_grads_fun[state](gradients)
        return predictions


    @tf.function
    def eval_step(self, samples):

        predictions = self.model(samples, training=False)

        return predictions

    def test_step(self, samples, targets):
        ''' generate the targets and apply the corresponding loss function '''
        predictions = self.model(samples, training=False)

        loss = [self.loss_fn[0](targets[..., 0, tf.newaxis], predictions[0]),
                self.loss_fn[1](targets[..., 1, tf.newaxis], predictions[1])]
        self.accumulator_ValidLoss[0](loss[0])
        self.accumulator_ValidLoss[1](loss[1])
        return predictions

    def train_epoch(self, epoch):
        self.accumulator_loss[0].reset_states()
        self.accumulator_loss[1].reset_states()
        self.accumulator_ValidLoss[0].reset_states()
        self.accumulator_ValidLoss[1].reset_states()


        for samples, targets in self.data['train']:
            predictions = self.train_step(samples, targets, self.state)

        for samples, targets in self.data['test']:
            predictions = self.test_step(samples, targets)


    def change_state(self, new_state):
        self.state = new_state
        if self.state == Wnet_vgg.UNET_BIN:
            self.model.layers[1].trainable = True
            self.model.layers[3].trainable = False
            self.optimizers[0] = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        if self.state == Wnet_vgg.UNET_MULTI:
            self.model.layers[1].trainable = False
            self.model.layers[3].trainable = True
            self.optimizers[1] = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        if self.state == Wnet_vgg.WNET:
            self.model.layers[1].trainable = True
            self.model.layers[3].trainable = True
            self.optimizers[2] = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.current_optimizer=self.optimizers[self.state]
        print(f'Changed State to {self.state}')



    def evaluate_train(self, epoch):
        self.model.reset_states()

        for samples, targets in self.data['train']:
            predictions = self.eval_step(samples)
        print(f'epoch: {epoch}, train_loss[0]: {self.accumulator_loss[0].result()}, train_loss[1]: {self.accumulator_loss[1].result()}, validation_loss[0]: {self.accumulator_ValidLoss[0].result()}, validation_loss[1]: {self.accumulator_ValidLoss[1].result()}')

        if(self.config['Plot_Results']):
            self.print_during_train(samples[0,..., 0], targets[0,..., 1],predictions[0][0, ..., 0], np.argmax(predictions[1][0, ...], axis=-1),fig_name=f"epoch: {epoch}")



    def print_during_train(self, CT_img, GT_mask, PRED_bin,Pred_multi, plot=1, save=0, has_pred=1, save_dir='', fig_name=''):
        plt.figure()
        plt.suptitle(fig_name)
        plt.subplot(1 + has_pred, 2, 1)  # plot 1
        plt.imshow(CT_img, cmap='gray')
        plt.title('MRI')
        plt.axis('off')
        plt.subplot(1 + has_pred, 2, 2)  # plot 2
        plt.imshow(CT_img, cmap='gray')
        plt.title('GT')
        plt.imshow(GT_mask, cmap='jet', alpha=0.4, interpolation='none',vmin=0,vmax=4)  # interpolation='none'
        plt.colorbar()
        plt.axis('off')
        if has_pred == 1:
            plt.subplot(1 + has_pred, 2, 3)  # plot 3
            plt.imshow(CT_img, cmap='gray')
            plt.title('Pred')
            plt.imshow(PRED_bin, cmap='jet', alpha=0.4, interpolation='none',vmin=0,vmax=1)  # interpolation='none'
            plt.colorbar()
            plt.axis('off')
            plt.subplot(1 + has_pred, 2, 4)  # plot 4   (for easier comparrison to both CT and GT)
            plt.imshow(CT_img, cmap='gray')
            plt.title('Pred')
            plt.imshow(Pred_multi, cmap='jet', alpha=0.4, interpolation='none',vmin=0,vmax=4)  # interpolation='none'
            plt.colorbar()
            plt.axis('off')
        if (plot == 1):
            plt.show()
        elif (save == 1):
            plt.savefig('../OUTPUT/{}{}'.format(save_dir, fig_name))


    def evaluate_test(self, epoch):
        self.model.reset_states()

        for samples, targets in self.data['train']:
            predictions = self.eval_step(samples)



    def save_manager(self, epoch):
        success=False
        path_save=''
        while(not success):
            try:
                path_save = self.ckpt_manager.save(checkpoint_number=self.global_epoch)
                success=True
            except:
                success=False
                time.sleep(0.5)
        return path_save

    def update_losses(self,epoch):
        self.train_loss[self.state][epoch, 0] = self.accumulator_loss[0].result()
        self.train_loss[self.state][epoch, 1] = self.accumulator_loss[1].result()
        self.valid_loss[self.state][epoch, 0] = self.accumulator_ValidLoss[0].result()
        self.valid_loss[self.state][epoch, 1] = self.accumulator_ValidLoss[1].result()

    def save_loss(self,epoch):
        df=[pd.DataFrame() for i in range(self.state)]
        for i in range(len(df)):
            df[i]['loss1']=self.train_loss[i][:,0]
            df[i]['loss2']=self.train_loss[i][:,1]
            df[i]['v_loss1']=self.valid_loss[i][:,0]
            df[i]['v_loss2']=self.valid_loss[i][:,1]
            df[i].to_csv(f"{self.config['results_dir']}{self.config['exp_name']}/Losses/{self.config['loss_log_filenames'][i]}")

    def save_config(self):
        my_dict=self.config.copy()
        my_dict.pop("train_writer",None)
        my_dict.pop("test_writer", None)
        for key in my_dict.keys():
            if(type(my_dict[key])==np.ndarray):
                my_dict[key]=my_dict[key].tolist()
        with open(f"{self.config['results_dir']}{self.config['exp_name']}/config.json", 'w') as outfile:
            json.dump(my_dict, outfile, separators=(",\n", ":"))

    def train(self):
        self.save_config()
        for state in [Wnet_vgg.UNET_BIN,Wnet_vgg.UNET_MULTI,Wnet_vgg.WNET]:
            self.change_state(state)
            print(f"num_epochs: {self.config['num_epochs']}")
            for epoch in range(self.config['num_epochs'][state]):

                self.train_epoch(epoch)
                self.update_losses(epoch)
                if epoch % self.config['eval_freq'] == 0:
                    self.evaluate_train(epoch)
                    self.evaluate_test(epoch)

                if epoch % self.config['ckpt_save_freq'] == 0:
                    self.save_manager(epoch)

                if epoch % self.config['save_loss_freq'] == 0:
                    self.save_loss(epoch)
                self.global_epoch.assign_add(1)

        self.save_manager(epoch)
        self.save_loss(epoch)



