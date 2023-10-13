import numpy as np
import zipjson
import json
import torch
from torch import nn
from pathlib import Path
from loguru import logger
from torch.utils.data import Dataset

class ContactsDataset(Dataset):
    def __init__(self, contacts_path, timestamps, mirror_contacts=False):
        super().__init__()
        self.mirror_contacts = mirror_contacts
        self.timestamps = timestamps

        contacts_path = Path(contacts_path)
        self.data_name = contacts_path.stem
        self.contacts_data = json.load(contacts_path.open())
        self.contacts_intervals = {surf: np.asarray(intervals) for surf, intervals in self.contacts_data['contacts'].items()}


    def __len__(self):
        return len(self.timestamps)

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
                closest_start_ind = ContactsDataset.find_closest_interval_before(intervals, timestamp)
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

    def get_contacts(self, idx):
        timestamp = self.timestamps[idx]
        contacts_dict = self.get_current_contacts(self.contacts_intervals, timestamp, self.mirror_contacts)
        contacts_tensors = {k: torch.tensor(v, dtype=torch.bool) for k, v in contacts_dict.items()}
        contacts_tensors = torch.stack([contacts_tensors[x] for x in ['left_hand', 'right_hand']], dim=0)
        return contacts_tensors

    def __getitem__(self, idx):
        contacts_tensors = self.get_contacts(idx)
        return contacts_tensors, {"name":self.data_name, "index": idx, "mirrored":self.mirror_contacts}


class SequenceContactsDataset(ContactsDataset):
    def __init__(self, contacts_path, timestamps, radius=0, *args, **kwargs):
        super().__init__(contacts_path, timestamps, *args, **kwargs)

        self.window_radius = radius
        self.window_size = radius * 2 + 1
        self.seqlen = len(timestamps) - self.window_size + 1

    def __len__(self):
        return self.seqlen

    def __getitem__(self, idx):
        contacts_tensors = self.get_contacts(idx + self.window_radius)
        return contacts_tensors, {"name":self.data_name, "index": idx + self.window_radius, "mirrored":self.mirror_contacts}
