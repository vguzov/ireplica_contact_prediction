import json
from collections import defaultdict

from contacts.dataset import PosesContactsDataset, SequencePosesContactsDataset
from contacts.model import SequenceContactClassifier, TransformerModel

import numpy as np
from loguru import logger
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
import pytorch_lightning as pl
from pathlib import Path
import toml
import dacite

from config import GeneralConfig

from argparse import ArgumentParser


def get_last_version_dir(dirpath: Path):
    max_vnum = None
    for path in dirpath.iterdir():
        if path.name.startswith("version_"):
            try:
                vnum = int(path.name[len("version_"):])
            except ValueError:
                logger.debug(f"{path.name} cannot be parsed as version folder")
            else:
                if max_vnum is None or max_vnum < vnum:
                    max_vnum = vnum
    return max_vnum


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path)
    parser.add_argument("-co", "--checkpoint_override", type=Path)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device)

    config_args_dict = toml.load(open(args.config))
    config_args = dacite.from_dict(data_class=GeneralConfig, data=config_args_dict, config=dacite.Config(cast=[Path]))

    input_dim = config_args.body_dim + 2 * config_args.hand_dim if config_args.include_hands else config_args.body_dim
    window_radius = config_args.arch.window_radius
    window_size = window_radius * 2 + 1

    logger.debug(f"Running with the following parameters:\n {config_args}")

    arch_args = config_args.arch

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

    test_sampler = SubsetRandomSampler(test_indices)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=config_args.batch_size,
                                              sampler=test_sampler, num_workers=config_args.dataloader_threads)

    if args.checkpoint_override is None:
        exp_root_dir = config_args.outdir / config_args.expname
        last_version = get_last_version_dir(exp_root_dir / "lightning_logs")
        checkpoint_dir = exp_root_dir / f"lightning_logs/version_{last_version}/checkpoints"
        checkpoint_path = list(checkpoint_dir.glob("model-best_acc*.ckpt"))[0]
    else:
        checkpoint_path = args.checkpoint_override
    logger.info(f"Loading checkpoint {checkpoint_path}")

    if arch_args.arch_name == "transformer":
        contact_classifier = SequenceContactClassifier.load_from_checkpoint(checkpoint_path,
                                                                            nn_model_logit=TransformerModel(input_dim, arch_args.internal_dim,
                                                                                                            window_size,
                                                                                                            arch_args.out_dim,
                                                                                                            arch_args.multiheadatt_heads_count,
                                                                                                            arch_args.linear_dim,
                                                                                                            arch_args.transformer_layers_count),
                                                                            lr=config_args.learning_rate)

    else:
        raise NotImplementedError(f"Unknown arch : {arch_args.arch_name}")

    contact_classifier.to(device)

    trainer = pl.Trainer()
    logger.info("Performing the test run")
    trainer.test(contact_classifier, test_loader)
    logger.info("Done")
