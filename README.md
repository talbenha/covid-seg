# A Deep Ensemble Learning Approach to Lung CT Segmentation for COVID-19 Severity Assessment

## _Training_
```sh
python main_train.py --config <config_path> --exp_name <exp_name> --run_name <run_name> --tag_name <tag_name> --batch_size <batch_size>
--num_epochs <epochs1 epochs2 epochs3> --seed <training_seed> --visible_gpu <gpu_num> --checkpoint_subdir <checkpoint_path> --log_subdir <log_path> 
--Starting_ckpt <num> --Weights_Cross_Entropy_Loss <w1 w2 w3 w4> --Weights_Dice_Loss <w1 w2 w3 w4> --Weights_Coeff <lambda> --quiet
```
## _Training variables_
**--config**

Path to the configuration file.

The configuration file contains most of the variables necessary for running - hyperparameters model, locally relative path, etc.
Please insert the following path in the config file to run training:

1. `data_path`: the path to the data directory
2. `train_folders`: a list contains 3 lists - \
a. The first list is of folders with raw CT data\
b. The second list is of folders with lung masks\
c. The third list is of folders with full masks\
*The three lists need to be of the same length and to correspond with each other, e.g., the i-th element of each list is a folder of CT, lung-segmentation & full segmentation of the same images.\
For example the following folders:
<pre>
data_path
|-- ct_folder_A
|   |-- file_1.nii
|   |-- file_2.nii
|   `-- file_3.nii
|-- lung_mask_folder_A
|   |-- file_1.nii
|   |-- file_2.nii
|   `-- file_3.nii
|-- full_mask_folder_A
|   |-- file_1.nii
|   |-- file_2.nii
|   `-- file_3.nii
|-- ct_folder_B
|   |-- file_1.nii
|   `-- file_2.nii
|-- lung_mask_folder_B
|   |-- file_1.nii
|   `-- file_2.nii
`-- full_mask_folder_B
    |-- file_1.nii
    `-- file_2.nii
</pre>
will be resolved into the list of lists:\
[["ct_folder_A", "ct_folder_B"],\
["lung_mask_folder_A", "lung_mask_folder_B"],\
["full_mask_folder_A", "full_mask_folder_B"]].\
Notice that within each folders triplet, the files are sorted correspondingly.
3.  `test_folders`: same as `train_folders` for the test files. If a segmentation does not exist, replace its corresponding value with null.
4. `results_dir`: the path to an output directory, all outputs will be saved in that directory under the experiment name.
5. `checkpoint_dir`: used for loading a trained model.
6. `Starting_ckpt`: if null - initializes from scratch, if -1 initializes the latest checkpoint, otherwise - initialize the specific checkpoint number provided (assuming the file exists).
7. `log_dir`: the path to the directory for logs.


**--exp_name**

Experiment name

**--run_name**

A run name for tensorboard usage

**--tag_name**

A tag name for tensorboard usage

**--batch_size**

Batch size in training

**--num_epochs**

Number of epochs in training

**--seed**

If not set, random seed

**--visible_gpu**

Set the available GPU for running

**--checkpoint_subdir**

Set the path to the subdirectory to save checkpoints

**--log_subdir**

set the path to the subdirectory to save logs

**--Starting_ckpt**

To continue training, Restores checkpoints, -1 for latest

**--Weights_Cross_Entropy_Loss**

Set Weights for CE loss, during runtime the weights are normalized to the sum of one

**--Weights_Dice_Loss**

Set Weights for Dice loss, during runtime the weights are normalized to the sum of one

**--Weights_Coeff**

Set Lambda Coeff between CE & Dice losses

**--quiet**

for debug usage, if the flag is set - use minimal debug outputs.

*All of the arguments' default values can be provided in the config file.
## _Evaluation using One Model_
```sh
python main_inference.py --config <config_path> --visible_gpu <gpu_num> --checkpoint_dir <checkpoint_path> --save_output <output_path>
```
## _Evaluation variables_
**--config**

Same as described in the training variables.

**--visible_gpu**

Set the available GPU for running

**--checkpoint_dir**

Set the path for the trained model checkpoints, load the last checkpoint

**--save_output**

Set the path for the output directory, save model prediction in the hierarchy of the test data path

## _Evaluation using Model Ensemble_
```sh
python main_inference.py --config <config_path> --visible_gpu <gpu_num> --checkpoint_dir <checkpoint_path> --ensemble_from <path1 path2 path3 ...> --save_output <output_path> --ensemble
```
## _Evaluation variables_
**--config**

Same as described in the training variables.

**--visible_gpu**

Set the available GPU for running

**--checkpoint_dir**

Mother directory for all ensemble experiments.

**--ensemble_from**

Set the path for the trained models' checkpoints used from the ensemble, load the last checkpoint. The mother directory is checkpoint_dir. Each directory of the experiment must contain a "Checkpoints" folder with the corresponding model checkpoints.

**--save_output**

Set the path for the output directory, save model prediction in the hierarchy of the test data path.

## _SUMC Data & Additional Annotations_
The SUMC data and additional annotations from the paper are provided under the tags of this repository - [Data](https://github.com/talbenha/covid-seg/releases/tag/additional-annotations)

## _Citation_
If you find either the code or the data useful for your research, cite our paper:
```sh
@inproceedings{,
title={A Deep Ensemble Learning Approach to Lung CT Segmentation for COVID-19 Severity Assessment},
author={Tal Ben-Haim and Ron Moshe Sofer and Gal Ben-Arie and Ilan Shelef and Tammy Riklin Raviv$^{1}$},
booktitle={IEEE International Conference on Image Processing (ICIP) 2022},
year={2022}
}
```
