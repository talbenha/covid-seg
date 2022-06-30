import numpy as np
import tensorflow as tf
from models.models import build_model
from utils.utils import  save_nifti
from utils.logger import create_dirs
from functools import partial
import os


def build_predicter(config):
    if config['trainer_name'] == "segmentation":
        predicter = SegPred(config)
    else:
        raise ValueError("'{}' is an invalid model name")

    return predicter


class SegPred:
    def __init__(self, config):
        self.config = config
        self.save_path = config["save_output"]

        if config["ensemble"]:
            self.model = None
            self.ens_pred_functions = []
            for name_ckpt in config["ensemble_from"]:
                checkpoint_directory = os.path.join(self.config['checkpoint_dir'], name_ckpt)
                checkpoint_directory = os.path.join(checkpoint_directory, "Checkpoints")

                self.ens_pred_functions.append(partial(self.predict_model, checkpoint_directory=checkpoint_directory))
        else:
            self.model = build_model(config)
            self.ens_pred_functions = None
            checkpoint_final = tf.train.Checkpoint(generator=self.model)
            checkpoint_directory = self.config['checkpoint_dir']
            checkpoint_name = tf.train.latest_checkpoint(checkpoint_directory)
            checkpoint_final.restore(checkpoint_name)
            if checkpoint_name:
                print(f'Restored checkpoint: {checkpoint_name.split("/")[-1]}')

    def predict_model(self, data, data_type, config, checkpoint_directory=None):
        if checkpoint_directory is None:
            raise ValueError('Checkpoint directory is null')
        tf.keras.backend.clear_session()
        model = build_model(config)
        checkpoint_final = tf.train.Checkpoint(generator=model)
        checkpoint_name = tf.train.latest_checkpoint(checkpoint_directory)
        checkpoint_final.restore(checkpoint_name)
        return self.give_pred(model,data,data_type)

    def give_pred(self,model, data, data_type):
        pred_mul_all = []
        for ind, (samples, _) in enumerate(data[data_type]):
            predictions = model(samples, training=False)
            pred_mul_all.append(predictions[1])
        pred_mul_all = np.concatenate(pred_mul_all, axis=0)

        return pred_mul_all

    def ensemble_preds(self, data, data_type):

        pred_mul_all_ensemble = []

        for pred_function in self.ens_pred_functions:
            pred_mul_all = pred_function(data,data_type,self.config)
            pred_mul_all_ensemble.append(pred_mul_all[np.newaxis,...])


        pred_mul_all_ensemble = np.concatenate(pred_mul_all_ensemble, axis=0)
        return pred_mul_all_ensemble


    def save_preds(self, data, data_type='test'):
        path = os.path.join(self.config['save_output'],self.config["test_folders"][0][0])
        if self.model is None:
            pred_mul_all_ensemble = self.ensemble_preds(data, data_type)
            pred_mul_avg = pred_mul_all_ensemble.mean(axis=0)
            pred_mul_hard = np.argmax(pred_mul_avg, axis=-1)
        else:
            pred_mul_all = self.give_pred(self.model, data, data_type)
            pred_mul_hard = np.argmax(pred_mul_all, axis=-1)
        create_dirs(path)
        save_nifti(pred_mul_hard, os.path.join(path, "prediction.nii.gz"))




