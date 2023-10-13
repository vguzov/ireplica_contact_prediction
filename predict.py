import os
import torch
import zipjson
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader

from contacts.dataset import SequencePosesContactsDataset
from contacts.model import SequenceContactClassifier, TransformerModel


def get_outpath(outdir: Path, seqname: str, expname: str, makedir=False):
    if expname is not None and len(expname) > 0:
        outpath = outdir / f"{seqname}/{expname}.json.zip"
        if makedir:
            (outdir / f"{seqname}").mkdir(exist_ok=True, parents=True)
    else:
        outpath = outdir / f"{seqname}.json.zip"
    return outpath


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--indir", type=Path, default=Path("./smpl_motions"), help="Directory with input poses")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("./predicted_contacts"), help="Output directory for predicted contacts")
    parser.add_argument("--prefilter", default="", help="Filter input files by this substring")
    parser.add_argument("--expname", default=None, help="Experiment name, if set, will be used expname as filename "
                                                        "(sequence name will be used as subdirectory)")
    parser.add_argument("--only_failed", action="store_true", help="Process only files that do not have output yet")
    parser.add_argument("-ch", "--checkpoint", type=Path, default=None, help="Path to model checkpoint")
    parser.add_argument("-ck", "--checkpoint_dir", type=Path,
                        default=Path("./checkpoints"), help="If checkpoint is not set, use this directory to search for best accuracy checkpoints")
    parser.add_argument("--device", default="cuda", help="Device to run the model on")
    parser.add_argument("-bs", "--batch_size", type=int, default=1000, help="Batch size for inference")
    parser.add_argument("-json", "--json_motion_format", action="store_true", help="Search for .json motion files instead of .json.zip")
    parser.add_argument("-nh", "--no_hands", action="store_false", default=True, dest="with_hands",
                        help="Don't include hand data in the input poses")
    parser.add_argument("--workers", type=int, default=24, help="Number of workers for data loading")

    args = parser.parse_args()

    if args.checkpoint is None:
        candidates = list(args.checkpoint_dir.glob("model-best_acc*.ckpt"))
        if len(candidates) == 0:
            candidates = args.checkpoint_dir.glob("*.ckpt")
        args.checkpoint = candidates[0]
        logger.info(f"Determined checkpoint: {args.checkpoint}")

    include_hands = args.with_hands

    input_candidates = sorted(args.indir.glob("*.json.zip" if not args.json_motion_format else "*.json"))

    device = torch.device(args.device)

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.only_failed:
        input_candidates = [x for x in input_candidates if
                            not get_outpath(args.outdir, os.path.splitext(x.name if args.json_motion_format else x.stem)[0], args.expname).is_file()]

    if args.prefilter is not None:
        input_list = [x for x in input_candidates if args.prefilter in str(x)]
    else:
        input_list = input_candidates

    logger.info(f"Loading model checkpoint: {args.checkpoint.name}")
    window_radius = 30
    window_size = window_radius * 2 + 1
    if include_hands:
        contact_classifier = SequenceContactClassifier.load_from_checkpoint(args.checkpoint,
                                                                            nn_model_logit=TransformerModel(69 + 2 * 45, 32, window_size, 2, 8, 32, 3))
    else:
        contact_classifier = SequenceContactClassifier.load_from_checkpoint(args.checkpoint,
                                                                            nn_model_logit=TransformerModel(69, 32, window_size, 2, 8, 32, 3))
    contact_classifier.to(device)

    surfaces = ["left_hand", "right_hand"]

    for ind, poses_path in enumerate(input_list):
        logger.info(f"Processing {poses_path.name} ({ind + 1}/{len(input_list)})")
        dataset = SequencePosesContactsDataset(poses_path, return_timestamps=True, radius=window_radius, include_hands=include_hands,
                                               old_input_pose_format=args.json_motion_format)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False)
        all_timestamps = []
        all_labels = []
        with torch.no_grad():
            for input_data, timestamps, contact_ts in tqdm(dataloader):
                input_data = input_data.to(device)
                predicted_labels = contact_classifier(input_data)
                all_timestamps.append(timestamps[:, window_radius].cpu().numpy())
                all_labels.append(predicted_labels.cpu().numpy())
        logger.info("Concatenating data")
        all_timestamps = np.concatenate(all_timestamps, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        resdict = {"version": 2, "type": "sequential", "sequence": {surfaces[i]: all_labels[:, i].tolist() for i in range(all_labels.shape[1])},
                   "timestamps": all_timestamps.tolist(),
                   "sequence_mintime": float(all_timestamps[0]), "sequence_maxtime": float(all_timestamps[-1])}
        logger.info("Saving to zipped json")
        outpath = get_outpath(args.outdir, os.path.splitext(poses_path.name if args.json_motion_format else poses_path.stem)[0], args.expname,
                              makedir=True)
        zipjson.dump(resdict, outpath.open("wb"))
    logger.info("Done")
