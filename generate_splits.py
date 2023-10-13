import numpy as np
import zipjson
from argparse import ArgumentParser
from pathlib import Path
import toml
import json
import dacite
from loguru import logger

from config import GeneralConfig


def place_new_cluster(cluster_size, existing_clusters, arr_size):
    def check_placement(cluster_center):
        curr_cluster = [cluster_center - cluster_halfsize, cluster_center + cluster_halfsize]
        for existing_cluster in existing_clusters:
            if not ((curr_cluster[0] < existing_cluster[0] and curr_cluster[1] < existing_cluster[0]) or (
                    curr_cluster[1] > existing_cluster[1] and curr_cluster[0] > existing_cluster[1])):
                return False
        return True

    cluster_halfsize = (cluster_size + 1) // 2
    cluster_center = None
    while cluster_center is None or not check_placement(cluster_center):
        cluster_center = np.random.randint(cluster_halfsize, arr_size - cluster_halfsize)
    return [cluster_center - cluster_halfsize, cluster_center + cluster_halfsize]


def clusters_to_inds(cluster_list, padding_size):
    inds = []
    for cluster in cluster_list:
        if cluster[1] - cluster[0] < 2 * padding_size:
            continue
        inds.append(np.arange(cluster[0] + padding_size, cluster[1] - padding_size))
    inds = np.concatenate(inds)
    return inds


parser = ArgumentParser()
parser.add_argument("-c", "--config", type=Path)
parser.add_argument("-cl", "--clusters", type=int, default=3)
parser.add_argument("-ts", "--test_split", type=float, default=0.05)
parser.add_argument("-vs", "--val_split", type=float, default=0.05)
parser.add_argument("-rng", "--seed", type=int, default=42)

args = parser.parse_args()

config_args_dict = toml.load(open(args.config))
config_args = dacite.from_dict(data_class=GeneralConfig, data=config_args_dict, config=dacite.Config(cast=[Path]))
data_args = config_args.dataset

window_radius = config_args.arch.window_radius
window_size = window_radius * 2 + 1

np.random.seed(args.seed)
data_args.splits_dir.mkdir(parents=True, exist_ok=True)

for seqname in data_args.sequences:
    logger.info(f"Processing {seqname}")
    poses_data = zipjson.load((data_args.poses_dir / (seqname + ".json.zip")).open("rb"))
    sequence = poses_data["sequence"]
    poses_len = len(sequence)
    seq_groups = ["train", "test", "valid"]
    res_inds = {x: np.array([]) for x in seq_groups}
    if seqname in data_args.no_train_sequences:
        seq_groups.remove("train")
    elif seqname in data_args.no_valid_sequences:
        seq_groups.remove("valid")
    elif seqname in data_args.no_test_sequences:
        seq_groups.remove("test")
    if len(seq_groups) == 0:
        logger.warning(f"Sequence {seqname} is not used in any split")
    elif len(seq_groups) == 1:
        res_inds[seq_groups[0]] = np.arange(poses_len - 2 * window_radius, dtype=np.int32)
    else:
        test_len = int(args.test_split * poses_len)
        test_cluster_size = test_len // args.clusters + 2 * window_radius  # 2*window_radius is a dead zone around each cluster
        windows_count = poses_len // window_size
        existing_clusters = []
        test_clusters = []
        for _ in range(args.clusters):
            cl = place_new_cluster(test_cluster_size, existing_clusters, poses_len)
            test_clusters.append(cl)
            existing_clusters.append(cl)
        test_clusters = sorted(test_clusters, key=lambda x: x[0])

        if len(seq_groups) == 3:
            val_len = int(args.val_split * poses_len)
            val_cluster_size = val_len // args.clusters + 2 * window_radius
            val_clusters = []
            for _ in range(args.clusters):
                cl = place_new_cluster(val_cluster_size, existing_clusters, poses_len)
                val_clusters.append(cl)
                existing_clusters.append(cl)
            val_clusters = sorted(val_clusters, key=lambda x: x[0])

        existing_clusters = sorted(existing_clusters, key=lambda x: x[0])

        last_end = -1
        train_clusters = []
        for cl in existing_clusters:
            if last_end + 1 < cl[0] - 1:
                train_clusters.append([last_end + 1, cl[0] - 1])
            last_end = cl[1]
        if existing_clusters[-1][1] < poses_len:
            train_clusters.append([existing_clusters[-1][1], poses_len])

        train_inds = clusters_to_inds(train_clusters, window_radius) - window_radius
        test_inds = clusters_to_inds(test_clusters, window_radius) - window_radius
        if len(seq_groups) == 3:
            val_inds = clusters_to_inds(val_clusters, window_radius) - window_radius
            res_inds.update({"train": train_inds, "valid": val_inds, "test": test_inds})
        else:  # len(seq_groups) == 2
            res_inds.update({seq_groups[0]: train_inds, seq_groups[1]: test_inds})

        for k, v in res_inds.items():
            if len(v) > 0:
                np.random.shuffle(v)

    res = {"window_radius": window_radius, "train": res_inds["train"].tolist(), "valid": res_inds["valid"].tolist(),
           "test": res_inds["test"].tolist()}
    all_size = poses_len - 2 * window_radius
    logger.info(
        f"Stats: train {len(res_inds['train']) / all_size * 100:.2f}%; test {len(res_inds['test']) / all_size * 100:.2f}%; valid {len(res_inds['valid']) / all_size * 100:.2f}%")
    logger.debug("Writting to json..")
    json.dump(res, (data_args.splits_dir / (seqname + ".json")).open("w"), indent=0)

logger.info("Done")
