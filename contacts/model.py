import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard.summary import pr_curve, compute_curve
from loguru import logger


class FramewiseContactClassifier(pl.LightningModule):
    def __init__(self, nn_model_logit, lr=1e-3):
        super().__init__()
        self.model = nn_model_logit
        self.lr = lr

    def training_step(self, batch, batch_idx):
        input_pose, output_contacts = batch
        predicted_contacts = self.model(input_pose)
        loss = nn.functional.binary_cross_entropy_with_logits(predicted_contacts, output_contacts.float())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_pose, output_contacts = batch
        predicted_contacts = self.model(input_pose)
        loss = nn.functional.binary_cross_entropy_with_logits(predicted_contacts, output_contacts.float())
        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x, *args, **kwargs):
        return torch.sigmoid(self.model(x))


first_nn_model_logit = nn.Sequential(nn.Linear(2 * 45 + 69, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                     nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))


class SequenceContactClassifier(pl.LightningModule):
    def __init__(self, nn_model_logit, lr=1e-3):
        super().__init__()
        self.model = nn_model_logit
        self.lr = lr
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        input_pose, output_contacts = batch
        predicted_contacts = self.model(input_pose)
        loss = nn.functional.binary_cross_entropy_with_logits(predicted_contacts, output_contacts.float())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_pose, output_contacts = batch
        predicted_contacts_logits = self.model(input_pose)
        loss = nn.functional.binary_cross_entropy_with_logits(predicted_contacts_logits, output_contacts.float())
        self.log("validation_loss", loss)
        predicted_contacts = torch.sigmoid(predicted_contacts_logits)
        self.validation_step_outputs.append({"pred": predicted_contacts.cpu(), "gt": output_contacts.cpu()})
        return loss

    def test_step(self, batch, batch_idx):
        input_pose, output_contacts = batch
        predicted_contacts_logits = self.model(input_pose)
        loss = nn.functional.binary_cross_entropy_with_logits(predicted_contacts_logits, output_contacts.float())
        self.log("test_loss", loss)
        predicted_contacts = torch.sigmoid(predicted_contacts_logits)
        self.test_step_outputs.append({"pred": predicted_contacts.cpu(), "gt": output_contacts.cpu()})
        return loss


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


    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        all_pred = torch.cat([output['pred'] for output in outputs], dim=0)
        all_gt = torch.cat([output['gt'] for output in outputs], dim=0)
        self.add_pr_curve_tensorboard("left_hand", 0, all_pred, all_gt)
        self.add_pr_curve_tensorboard("right_hand", 1, all_pred, all_gt)
        self.log("LH accuracy@0.5", (((all_pred[:, 0] > 0.5) == all_gt[:, 0]).float()).mean())
        self.log("RH accuracy@0.5", (((all_pred[:, 1] > 0.5) == all_gt[:, 1]).float()).mean())
        self.log("accuracy_05", (((all_pred > 0.5) == all_gt).float()).mean())

    def on_test_epoch_end(self) -> None:
        outputs = self.test_step_outputs
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

        logger.info(f"TEST:  {ap:.3f} & {precision_at_thresh:.3f} & {recall_at_thresh:.3f} & {test_acc:.3f}")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x, *args, **kwargs):
        return torch.sigmoid(self.model(x))



class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  # [:d_model//2]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, item_size, internal_dim_size, seq_size, output_size, multiheadatt_heads_count, linear_dim, transformer_layers_count,
            dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(internal_dim_size, dropout)
        encoder_layers = TransformerEncoderLayer(internal_dim_size, multiheadatt_heads_count, linear_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, transformer_layers_count)
        self.encoder = nn.Linear(item_size, internal_dim_size)
        self.ninp = internal_dim_size
        self.decoder = nn.Sequential(nn.Linear(internal_dim_size, output_size), nn.ReLU())
        self.aggregator = nn.Linear(seq_size * output_size, output_size)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.aggregator.bias)
        nn.init.uniform_(self.aggregator.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        self.src_mask = None
        batch_dim = src.size(0)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        output = self.aggregator(output.view(batch_dim, -1))
        return output


class GRUModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, item_size, internal_dim_size, seq_size, output_size, multiheadatt_heads_count, linear_dim, transformer_layers_count,
            dropout=0.5):
        super(GRUModel, self).__init__()
        self.model_type = 'GRU'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(internal_dim_size, dropout)
        self.transformer_encoder = nn.GRU(internal_dim_size, internal_dim_size)
        self.encoder = nn.Linear(item_size, internal_dim_size)
        self.ninp = internal_dim_size
        self.decoder = nn.Sequential(nn.Linear(internal_dim_size, output_size), nn.ReLU())
        self.aggregator = nn.Linear(seq_size * output_size, output_size)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.aggregator.bias)
        nn.init.uniform_(self.aggregator.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        self.src_mask = None
        batch_dim = src.size(0)
        src = self.encoder(src) * math.sqrt(self.ninp)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        output = self.aggregator(output.view(batch_dim, -1))
        return output
