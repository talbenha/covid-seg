import argparse
import logging
import os
import glob
import shutil
import sys
import tensorflow as tf
from utils.config import read_config
from utils.logger import set_logger_and_tracker
import SimpleITK as sitk

logger = logging.getLogger("logger")



def get_args():
    """" collects command line arguments """

    argparser = argparse.ArgumentParser(description=__doc__)
    #train args
    argparser.add_argument('--config',                      default="configs/covid19.json", type=str, help='configuration file')
    argparser.add_argument('--exp_name',                    default=None, type=str, help='experiment name')
    argparser.add_argument('--run_name',                    default=None, type=str, help='run name')
    argparser.add_argument('--tag_name',                    default=None, type=str, help='tag name')
    argparser.add_argument('--batch_size',                  default=None, type=int, help='batch size in training')
    argparser.add_argument('--seed',                        default=None, type=int, help='randomization seed')
    argparser.add_argument('--visible_gpu',                 default="0" , type=str, help='sets visible gpu')
    argparser.add_argument('--checkpoint_subdir',           default=None  , type=str, help='subdir to save checkpoints')
    argparser.add_argument('--log_subdir',                  default=None  , type=str, help='subdir to save logs')
    argparser.add_argument('--num_epochs',                  default=None  , type=int, nargs=3, help='number of epochs in each state')
    argparser.add_argument('--Starting_ckpt',               default=None  , type=int, help='Restores checkpoints, -1 for latest')
    argparser.add_argument('--Weights_Cross_Entropy_Loss',  default=None  , type=float, nargs=5, help='Weights for CE loss')
    argparser.add_argument('--Weights_Dice_Loss',           default=None  , type=float, nargs=5, help='Weights for Dice loss')
    argparser.add_argument('--Weights_Coeff',               default=None  , type=float, help='Lambda Coeff between CE & Dice losses')

    argparser.add_argument('--quiet',                       dest='quiet', action='store_true')

    # inference 1 model args
    argparser.add_argument('--checkpoint_dir',              default=None, type=str, help='directory to load checkpoints')
    argparser.add_argument('--save_output',                 default=None, type=str, help='directory to output folder')


    # inference ensembles model args
    argparser.add_argument('--ensemble',                    dest='ensemble', action='store_true')
    argparser.add_argument('--ensemble_from',                default=None, type=str, nargs="+", help='please let the exp names')

    args = argparser.parse_args()
    return args

def gpu_init():
    """ Allows GPU memory growth """


    gpus = tf.config.experimental.list_physical_devices('GPU')
    logger.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.info("MESSAGE", e)

def save_scripts(config):
    if config['save_scripts']==0:
        return
    path = os.path.join(config['tensor_board_dir'], 'scripts')
    if not os.path.exists(path):
        os.makedirs(path)
    scripts_to_save = glob.glob('./**/*.py', recursive=True) + [config.config]
    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_file = os.path.join(path, os.path.basename(script))
            shutil.copyfile(os.path.join(os.path.dirname(sys.argv[0]),script), dst_file)

def preprocess_meta_data():
    """ preprocess the config for specific run:
            1. reads command line arguments
            2. updates the config file and set gpu config
            3. configure gpu settings
            4. Define logger
            5. Save scripts
    """

    args = get_args()

    config = read_config(args)

    gpu_init()

    set_logger_and_tracker(config)

    save_scripts(config)

    return config

def save_nifti(data,full_path):
    itk_img = sitk.GetImageFromArray(data)
    sitk.WriteImage(itk_img, full_path)



