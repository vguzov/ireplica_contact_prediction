import json
from collections import defaultdict

from contacts.dataset import SequencePosesContactsDataset
from contacts.model import SequenceContactClassifier, TransformerModel

import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import toml
import dacite

from config import GeneralConfig

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-c", "--config", type=Path)

args = parser.parse_args()

config_args_dict = toml.load(open(args.config))
config_args = dacite.from_dict(data_class=GeneralConfig, data=config_args_dict, config=dacite.Config(cast=[Path]))

input_dim = config_args.body_dim + 2 * config_args.hand_dim if config_args.include_hands else config_args.body_dim
window_radius = config_args.arch.window_radius
window_size = window_radius * 2 + 1

logger.debug(f"Running with the following parameters:\n {config_args}")

# Build the classifier

arch_args = config_args.arch

if arch_args.arch_name == "transformer":
    contact_classifier = SequenceContactClassifier(TransformerModel(input_dim, arch_args.internal_dim, window_size,
                                                                    arch_args.out_dim, arch_args.multiheadatt_heads_count,
                                                                    arch_args.linear_dim, arch_args.transformer_layers_count),
                                                   lr=config_args.learning_rate)
else:
    raise NotImplementedError(f"Unknown arch : {arch_args.arch_name}")

# Building the dataset

data_args = config_args.dataset
datasets_list = []
split_inds_list = defaultdict(list)
curr_offset = 0
for seqname in data_args.sequences:
    curr_dataset = SequencePosesContactsDataset(data_args.poses_dir / (seqname + ".json.zip"), data_args.contacts_dir / (seqname + ".json"),
                                                window_radius,
                                                include_hands=config_args.include_hands, mirror_contacts=False)
    curr_dataset_size = len(curr_dataset)
    curr_split_dict = json.load((data_args.splits_dir / (seqname + ".json")).open())
    datasets_list.append(curr_dataset)
    # Add split indices with the corresponding offset
    for part in ["train", "valid", "test"]:
        split_inds_list[part].append(np.asarray(curr_split_dict[part]) + curr_offset)
    # Update the index offset
    curr_offset += curr_dataset_size
    if data_args.include_mirrored_poses:
        curr_dataset_mirrored = SequencePosesContactsDataset(data_args.poses_dir / Path("mirrored") / (seqname + ".json.zip"),
                                                             data_args.contacts_dir / (seqname + ".json"), window_radius,
                                                             include_hands=config_args.include_hands, mirror_contacts=True)
        datasets_list.append(curr_dataset_mirrored)
        # Add the same indices for mirrored dataset (mind the updated index offset)
        for part in ["train", "valid", "test"]:
            split_inds_list[part].append(np.asarray(curr_split_dict[part]) + curr_offset)
        # Update the index offset
        curr_offset += curr_dataset_size

dataset = ConcatDataset(datasets_list)
split_inds = {part: np.concatenate(split_inds_list[part]) for part in ["train", "valid", "test"]}
train_indices, val_indices, test_indices = split_inds["train"], split_inds["valid"], split_inds["test"]

# In case there is a faulty dtype, cast to int64
train_indices = train_indices.astype(np.int64)
val_indices = val_indices.astype(np.int64)
test_indices = test_indices.astype(np.int64)

logger.debug(f"{train_indices.min()}, {train_indices.max()}, {len(dataset)}")
logger.debug(f"{val_indices.min()}, {val_indices.max()}, {len(dataset)}")
logger.debug(f"{test_indices.min()}, {test_indices.max()}, {len(dataset)}")

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=config_args.batch_size,
                          sampler=train_sampler, num_workers=config_args.dataloader_threads)
validation_loader = DataLoader(dataset, batch_size=config_args.batch_size,
                               sampler=val_sampler, num_workers=config_args.dataloader_threads)
test_loader = DataLoader(dataset, batch_size=config_args.batch_size,
                         sampler=test_sampler, num_workers=config_args.dataloader_threads)

checkpoint_outdir = config_args.outdir / config_args.expname
checkpoint_outdir.mkdir(parents=True, exist_ok=True)

checkpoint_callback_best = ModelCheckpoint(
    save_top_k=1,
    monitor="validation_loss",
    mode="min",
    filename="model-best-{epoch:02d}-{validation_loss:.2f}",
)

checkpoint_callback_best_acc = ModelCheckpoint(
    save_top_k=1,
    monitor="accuracy_05",
    mode="max",
    filename="model-best_acc-{epoch:02d}-{accuracy_05:.2f}",
)

checkpoint_callback_last = ModelCheckpoint(
    save_top_k=1,
    filename="model-last-{epoch:02d}-{validation_loss:.2f}",
)

logger.info(f"Saving to {checkpoint_outdir}")
trainer = pl.Trainer(max_epochs=config_args.max_epochs, accelerator="gpu",
                     default_root_dir=str(checkpoint_outdir),
                     callbacks=[checkpoint_callback_best, checkpoint_callback_last, checkpoint_callback_best_acc])
toml.dump(config_args_dict, (checkpoint_outdir / "config.toml").open("w"))
trainer.fit(contact_classifier, train_loader, val_dataloaders=validation_loader)
logger.info("Performing the test run")
trainer.test(contact_classifier, test_loader)
logger.info("Done")
