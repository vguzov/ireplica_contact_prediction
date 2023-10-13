from dataclasses import dataclass

import dataclasses

from pathlib import Path

from typing import Union, Optional, List, Dict


@dataclass
class ArchConfig:
    arch_name: str


@dataclass
class TransformerArchConfig(ArchConfig):
    window_radius: int = 30
    out_dim: int = 2
    multiheadatt_heads_count: int = 8
    linear_dim: int = 32
    internal_dim: int = 32
    transformer_layers_count: int = 3
    arch_name: str = "transformer"


@dataclass
class DatasetConfig:
    sequences: List[str]
    no_train_sequences: List[str]
    no_valid_sequences: List[str]
    no_test_sequences: List[str]
    contacts_dir: Path
    poses_dir: Path
    splits_dir: Path
    include_mirrored_poses: bool = True


@dataclass
class GeneralConfig:
    arch: TransformerArchConfig
    dataset: DatasetConfig
    outdir: Path
    expname: str
    dataloader_threads: int = 10
    include_hands: bool = True
    body_dim: int = 69
    hand_dim: int = 45
    max_epochs: int = 200
    batch_size: int = 100
    learning_rate: float = 1e-3


def config_to_dict(config: GeneralConfig):
    res_dict = {}
    config_dict = dataclasses.asdict(config)
    for param_name, param_val in config_dict.items():
        if isinstance(param_val, Path):
            param_val = str(param_val)
        res_dict[param_name] = param_val
    return res_dict
