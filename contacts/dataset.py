import numpy as np
import zipjson
import json
import torch
from torch import nn
from pathlib import Path
from loguru import logger
from torch.utils.data import Dataset


class PosesContactsDataset(Dataset):
    def __init__(self, poses_path, contacts_path=None, return_timestamps=False, include_hands=True, old_input_pose_format=False, mirror_contacts=False):
        super().__init__()
        poses_path = Path(poses_path)
        self.include_hands = include_hands
        self.return_timestamps = return_timestamps
        self.old_input_pose_format = old_input_pose_format
        self.mirror_contacts = mirror_contacts

        if poses_path.suffix == '.zip':
            self.poses_data = zipjson.load(poses_path.open("rb"))
        else:
            self.poses_data = json.load(poses_path.open())
        if self.old_input_pose_format:
            self.poses_seq = self.poses_data
        else:
            self.poses_seq = self.poses_data["sequence"]

        if contacts_path is not None:
            contacts_path = Path(contacts_path)
            self.contacts_data = json.load(contacts_path.open())
            self.contacts_intervals = {surf: np.asarray(intervals) for surf, intervals in self.contacts_data['contacts'].items()}
        else:
            self.contacts_data = None
        self.included_params = ['body_pose']
        if self.include_hands:
            if not self.old_input_pose_format:
                self.included_params += ['left_hand_pose', 'right_hand_pose']
            else:
                logger.warning("include_hands doesn't work together with old_input_pose_format; setting include_hands to False")
                self.include_hands = False

    def __len__(self):
        return len(self.poses_seq)

    @staticmethod
    def find_closest_interval_before(intervals, timestamp):
        start_diff = timestamp - intervals[:, 0]
        starts_happened_before = start_diff >= 0
        if np.count_nonzero(starts_happened_before) == 0:
            return None
        closest_start_ind = np.argmin(start_diff[starts_happened_before])
        return closest_start_ind

    @staticmethod
    def get_current_contacts(contacts_intervals, timestamp, mirror_contacts:bool):
        res = {}
        for surface_name, intervals in contacts_intervals.items():
            if len(intervals) == 0:
                res[surface_name] = False
            else:
                closest_start_ind = PosesContactsDataset.find_closest_interval_before(intervals, timestamp)
                if closest_start_ind is None:
                    res[surface_name] = False
                else:
                    contact_state = ((intervals[closest_start_ind, 1] - intervals[closest_start_ind, 0]) < 0) or (
                            intervals[closest_start_ind, 1] > timestamp)
                    res[surface_name] = contact_state
        if mirror_contacts:
            oldres = res
            res = {"left_hand": oldres["right_hand"], "right_hand": oldres["left_hand"]}
        return res

    def process_param(self, param_name, param_val):
        if param_name == "body_pose":
            if len(param_val) < 69:
                param_val = np.concatenate([param_val, np.zeros(69-len(param_val))])
        return param_val

    def get_pose(self, idx):
        pose_dict = self.poses_seq[idx]
        if self.old_input_pose_format:
            pose_tensors = torch.tensor(pose_dict['pose'][3:], dtype=torch.float32)
        else:
            pose_tensors = {k: torch.tensor(self.process_param(k,pose_dict[k]), dtype=torch.float32) for k in self.included_params}
            pose_tensors = torch.cat([pose_tensors[x] for x in self.included_params], dim=0)
        return pose_tensors

    def get_contacts(self, idx):
        timestamp = self.poses_seq[idx]['time']
        contacts_dict = self.get_current_contacts(self.contacts_intervals, timestamp, self.mirror_contacts)
        contacts_tensors = {k: torch.tensor(v, dtype=torch.bool) for k, v in contacts_dict.items()}
        contacts_tensors = torch.stack([contacts_tensors[x] for x in ['left_hand', 'right_hand']], dim=0)
        return contacts_tensors

    def __getitem__(self, idx):
        pose_tensors = self.get_pose(idx)
        if self.contacts_data is None and not self.return_timestamps:
            return pose_tensors
        output = [pose_tensors]
        if self.contacts_data is not None:
            contacts_tensors = self.get_contacts(idx)
            output.append(contacts_tensors)
        if self.return_timestamps:
            timestamp = self.poses_seq[idx]['time']
            output.append(torch.tensor(timestamp, dtype=torch.float32))
        return output


class SequencePosesContactsDataset(PosesContactsDataset):
    def __init__(self, poses_path, contacts_path=None, radius=0, *args, **kwargs):
        super().__init__(poses_path, contacts_path, *args, **kwargs)
        self.window_radius = radius
        self.window_size = radius * 2 + 1
        self.seqlen = len(self.poses_seq) - self.window_size + 1

    def __len__(self):
        return self.seqlen

    def __getitem__(self, idx):
        output_poses = torch.stack([self.get_pose(i) for i in range(idx, idx + self.window_size)], dim=0)
        if self.contacts_data is None and not self.return_timestamps:
            return output_poses
        output = [output_poses]
        if self.contacts_data is not None:
            contacts_tensors = self.get_contacts(idx + self.window_radius)
            output.append(contacts_tensors)
        if self.return_timestamps:
            timestamps = torch.tensor([self.poses_seq[i]['time'] for i in range(idx, idx + self.window_size)], dtype=torch.float32)
            output.append(timestamps)
            contact_ts = torch.tensor([self.poses_seq[idx + self.window_radius]['time']])
            output.append(contact_ts)
        return output
