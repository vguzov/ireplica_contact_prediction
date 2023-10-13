import numpy as np
import json
import zipjson
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm

MIRRORED_JOINTS = np.asarray([0, 3, 6, 9, 12, 15])

MIRRORED_INDS = (MIRRORED_JOINTS[:, np.newaxis]*3+np.arange(3)[np.newaxis, :]).flatten()
# logger.info(MIRRORED_INDS)

SWAPPED_JOINTS = np.asarray([
    (1, 2),
    (4, 5),
    (7, 8),
    (10, 11),
    (13, 14),
    (16, 17),
    (18, 19),
    (20, 21),
    (22, 23)
])
SWAPPED_INDS = (SWAPPED_JOINTS[:, np.newaxis, :]*3+np.arange(3)[np.newaxis, :, np.newaxis]).reshape(-1,2)

# X-axis mirroring
MIRROR_MTX = np.asarray([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=np.float64)

MODIFIED_FIELDS = ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose"]


def mirror_rotvec(rotvec_batch):
    rots = Rotation.from_rotvec(rotvec_batch)
    rotmats = rots.as_matrix()
    mirrored_rotmats = np.matmul(MIRROR_MTX,np.matmul(rotmats, MIRROR_MTX))
    mirrored_rotvecs = Rotation.from_matrix(mirrored_rotmats).as_rotvec()
    return mirrored_rotvecs


def mirror_body(body_pose_vct: np.ndarray):
    mirrored_vct = np.zeros_like(body_pose_vct)
    batch_size = body_pose_vct.shape[0]
    mirrored_vct[:, MIRRORED_INDS] = body_pose_vct[:, MIRRORED_INDS]
    mirrored_vct[:, SWAPPED_INDS[:, 0]] = body_pose_vct[:, SWAPPED_INDS[:, 1]]
    mirrored_vct[:, SWAPPED_INDS[:, 1]] = body_pose_vct[:, SWAPPED_INDS[:, 0]]
    mirrored_vct = mirror_rotvec(mirrored_vct.reshape(-1, 3)).reshape(batch_size, 72)
    return mirrored_vct


def mirror_pose(pose_dict):
    body_vct = np.concatenate([np.asarray(pose_dict["global_orient"]), np.asarray(pose_dict["body_pose"])])
    mirrored_vct = mirror_body(body_vct[np.newaxis, :])[0]
    mirrored_dict = {"global_orient": mirrored_vct[:3].tolist(), "body_pose": mirrored_vct[3:].tolist(),
                     "left_hand_pose": mirror_rotvec(np.asarray(pose_dict["right_hand_pose"]).reshape(-1,3)).flatten().tolist(),
                     "right_hand_pose": mirror_rotvec(np.asarray(pose_dict["left_hand_pose"]).reshape(-1,3)).flatten().tolist()}
    for k in pose_dict.keys():
        if k not in MODIFIED_FIELDS:
            mirrored_dict[k] = pose_dict[k]
    return mirrored_dict

def mirror_seq(pose_sequence):
    mirrored_pose_seq = []
    for pose_dict in tqdm(pose_sequence):
        mirrored_pose_seq.append(mirror_pose(pose_dict))
    return mirrored_pose_seq


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--indir", type=Path, default=Path("./data/body"))
    parser.add_argument("-o", "--outdir", type=Path, default=Path("./data/body/mirrored"))
    parser.add_argument("--prefilter", default="")
    parser.add_argument("--only_failed", action="store_true")

    args = parser.parse_args()

    input_candidates = sorted(args.indir.glob("*.json.zip"))

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.only_failed:
        input_candidates = [x for x in input_candidates if not (args.outdir / x.name).is_file()]

    if args.prefilter is not None:
        input_list = [x for x in input_candidates if args.prefilter in str(x)]
    else:
        input_list = input_candidates

    for ind, poses_path in enumerate(input_list):
        logger.info(f"Processing {poses_path.name} ({ind + 1}/{len(input_list)})")
        poses_data = zipjson.load(poses_path.open("rb"))
        poses_data["sequence"] = mirror_seq(poses_data["sequence"])
        logger.info("Saving to zipped json")
        zipjson.dump(poses_data, (args.outdir / poses_path.name).open("wb"))
