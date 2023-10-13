# iReplica – Contact Prediction Network
This is a pose-based contact prediction network used in 
["Interaction Replica: Tracking human–object interaction and scene changes from human motion"](https://virtualhumans.mpi-inf.mpg.de/ireplica).

This repository is part of the iReplica project, for more information and other modules please refer to the 
[core module repository](https://github.com/vguzov/ireplica).

## Installation

1. Clone this repo
2. Make sure pytorch is installed (follow instruction at https://pytorch.org/)
3. Install the dependencies with `pip install -r requirements.txt`.

## Pretrained models

The model trained on the H-Contact dataset are available here

| Model pretrain type         | Link                                                                      | Config                     | Train/val/test split (unpack it) |
|-----------------------------|---------------------------------------------------------------------------|----------------------------|----------------------------------|
| Doors (for hinged objects)  | [Download](https://nextcloud.mpi-klsb.mpg.de/index.php/s/KYqA7mCfZndoFaC) | [Link](configs/doors.toml) | [Link](splits/doors.tar.gz)      |
| Sofas (for sliding objects) | [Download](https://nextcloud.mpi-klsb.mpg.de/index.php/s/T3giJm9skcpnKyx) | [Link](configs/sofas.toml) | [Link](splits/sofas.tar.gz)      |

## Inference

To predict contacts from motion, run

```bash
python predict.py -i <input_motions_dir> -o <output_dir> -ch <checkpoint_path> [-nh] [-json] [-bs <batch_size>]
```

where `<input_motions_dir>` is the directory with `.json.zip` files containing the body poses,
`<output_dir>` is the directory where the predictions will be saved, `<checkpoint_path>` is the path to the model checkpoint,
`-nh` flag to use model without hands, `-json` searches for motion in `.json` file format instead of default `.json.zip`,
and `-bs` sets the batch size.

Input and output formats are described here:
<details>
  <summary>IO format description</summary>


#### Input format
Input is a `.json.zip` file with body poses in SMPL-H format. Each file should contain a dictionary with the following keys:
- `sequence`: list of body poses, each pose is a dictionary with the following keys:
    - `time`: timestamp of the pose in seconds
    - `global_orient`: global orientation of the body, axis-angle representation
    - `body_pose`: vector of SMPL/SMPL-H body pose parameters
    - `transl` (optional): global translation of the body root joint
    - `left_hand_pose` (if hands data is present): vector of SMPL-H left hand pose parameters
    - `right_hand_pose` (if hands data is present): vector of SMPL-H right hand pose parameters
- (Not required for contact prediction) `global`: global parameters of the body as a dictionary with the following keys:
    - `betas`: SMPL body shape parameters (in _SMPL_ blendspace, not in SMPL-H)
    - `gender`: SMPL model gender (male, female or neutral)

Additionally, the script also supports older motion format stored in `.json` files (some of the motions in EgoHOI dataset). 
The format is a list of dicts for each frame with the following keys:
- `time`: timestamp of the pose in seconds
- `pose`: SMPL body pose parameters, global and local orientations are concatenated
- `shape`: SMPL body shape parameters
- `translation`: global translation of the body root joint

#### Output format
Output is a `.json.zip` file with the following structure:
- `version`: version of the output format
- `type`: type of the prediction, sequential or interval-based, this framework uses only sequential
- `sequence`: predictions for each hand (dict with `left_hand` and `right_hand` keys), each prediction is a per-frame list of contact
  probabilities
- `timestamps`: list of timestamps for each frame in seconds
- `sequence_mintime`: start of sequence (minimum timestamp of the sequence in seconds)
- `sequence_maxtime`: end of sequence (maximum timestamp of the sequence in seconds)
</details>

## Training the network

### Config file

To create a config, check [configs/example.toml](configs/example.toml) for an example with commentaries on each parameter.

### Split generation

If needed, generate train/test/validation splits with

```bash
python generate_splits.py -c <config_file>
```

where `<config_file>` is the path to the config file. In the config file, specify the path where the splits will be saved by setting `dataset.splits_dir`

### Data augmentation

Augment the data by mirroring body poses with 
```bash
mirror_poses.py -i <input_poses_dir> -o <mirrored poses output dir>
```

### Training

To train, run

```bash
python train.py -c <config_file>
```

where `<config_file>` is the path to the config file.
Checkpoints and Tensorboard logs will be saved in the `output_dir` specified in the config file.

To evaluate the model, run

```bash
python test.py -c <config_file>
```

where `<config_file>` is the path to the config file.

## Dataset

The **H-Contact** dataset used to train this network for iReplica method is available at https://virtualhumans.mpi-inf.mpg.de/ireplica/datasets.html


## Citation

If you use this code, please cite our paper:

```
@inproceedings{guzov23ireplica,
    title = {Interaction Replica: Tracking human–object interaction and scene changes from human motion},
    author = {Guzov, Vladimir and Chibane, Julian and Marin, Riccardo and He, Yannan and Saracoglu, Yunus and Sattler, Torsten and Pons-Moll, Gerard},
    booktitle = {arXiv},
    year = {2023}}
```
