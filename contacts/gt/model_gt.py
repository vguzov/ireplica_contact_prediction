import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard.summary import pr_curve, compute_curve
from typing import Dict
from loguru import logger

class PrecomputedSequenceContactClassifier(pl.LightningModule):
    def __init__(self, answers_database:Dict[str, Dict[bool, np.ndarray]]):
        super().__init__()
        self.answers_database = answers_database

    def test_step(self, batch, batch_idx):
        output_contacts, data_info_batch = batch
        predicted_contacts_list = []
        for ind in range(output_contacts.size(0)):
            p = self.answers_database[data_info_batch["name"][ind]][data_info_batch["mirrored"][ind].item()][data_info_batch["index"][ind].item()]
            predicted_contacts_list.append(p)
        predicted_contacts = torch.tensor(np.stack(predicted_contacts_list, axis=0), dtype=output_contacts.dtype)
        # return loss
        return {"pred": predicted_contacts, "gt": output_contacts.cpu()}

    def add_pr_curve_tensorboard(self, class_name, class_index, pred, gt):
        tensorboard_truth = gt[:, class_index]
        tensorboard_probs = pred[:, class_index]

        writer = self.logger.experiment
        writer.add_pr_curve(class_name,
                            tensorboard_truth,
                            tensorboard_probs,
                            global_step=self.current_epoch)

    def compute_pr_metrics(self, class_index, pred, gt, logging_thresh = 0.5):
        tensorboard_truth = gt[:, class_index].cpu().numpy()
        tensorboard_probs = pred[:, class_index].cpu().numpy()
        num_thresh = 127
        logging_thresh_ind = int(num_thresh*logging_thresh)
        tp, fp, tn, fn, precision, recall = compute_curve(tensorboard_truth, tensorboard_probs, num_thresholds=num_thresh)
        ap = np.sum((recall[:-1] - recall[1:]) * precision[:-1])
        precision_at_thresh = precision[logging_thresh_ind]
        recall_at_thresh = recall[logging_thresh_ind]
        return ap, precision_at_thresh, recall_at_thresh


    def validation_epoch_end(self, outputs) -> None:
        all_pred = torch.cat([output['pred'] for output in outputs], dim=0)
        all_gt = torch.cat([output['gt'] for output in outputs], dim=0)
        self.add_pr_curve_tensorboard("left_hand", 0, all_pred, all_gt)
        self.add_pr_curve_tensorboard("right_hand", 1, all_pred, all_gt)
        self.log("LH accuracy@0.5", (((all_pred[:, 0] > 0.5) == all_gt[:, 0]).float()).mean())
        self.log("RH accuracy@0.5", (((all_pred[:, 1] > 0.5) == all_gt[:, 1]).float()).mean())
        self.log("accuracy_05", (((all_pred > 0.5) == all_gt).float()).mean())

    def test_epoch_end(self, outputs) -> None:
        all_pred = torch.cat([output['pred'] for output in outputs], dim=0)
        all_gt = torch.cat([output['gt'] for output in outputs], dim=0)
        self.add_pr_curve_tensorboard("test_left_hand", 0, all_pred, all_gt)
        self.add_pr_curve_tensorboard("test_right_hand", 1, all_pred, all_gt)
        self.log("TEST LH accuracy@0.5", (((all_pred[:, 0] > 0.5) == all_gt[:, 0]).float()).mean())
        self.log("TEST RH accuracy@0.5", (((all_pred[:, 1] > 0.5) == all_gt[:, 1]).float()).mean())
        test_acc = (((all_pred > 0.5) == all_gt).float()).mean()
        self.log("test_accuracy_05", test_acc)
        thresh = 0.5
        ap_lh, precision_at_thresh_lh, recall_at_thresh_lh = self.compute_pr_metrics(0, all_pred, all_gt, logging_thresh=thresh)
        ap_rh, precision_at_thresh_rh, recall_at_thresh_rh = self.compute_pr_metrics(1, all_pred, all_gt)
        ap = (ap_lh+ap_rh)/2.
        precision_at_thresh = (precision_at_thresh_lh+precision_at_thresh_rh)/2.
        recall_at_thresh = (recall_at_thresh_lh+ recall_at_thresh_rh)/2.
        self.log("TEST AP", ap)
        self.log("TEST Precision@0.5", precision_at_thresh)
        self.log("TEST Recall@0.5", recall_at_thresh)

        logger.info(f"TEST:  {ap:.3f} & {precision_at_thresh:.3f} & {recall_at_thresh:.3f} & {test_acc:.3f} \\\\")