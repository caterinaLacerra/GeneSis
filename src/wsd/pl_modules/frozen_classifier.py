from typing import Optional, List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig


def get_vocabulary_size(vocabulary_name: str):
    raise NotImplementedError


class WSDTransformer(pl.LightningModule):

    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)

        self.accuracy = pl.metrics.Accuracy()

        self.encoder = AutoModel.from_pretrained(conf.generative_model.transformer_name)
        transformer_hs = AutoConfig.from_pretrained(conf.generative_model.transformer_name).hidden_size
        self.classifier = nn.Linear(transformer_hs, get_vocabulary_size(conf.inventory.inventory_name))

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                instances_offsets: List[List[Tuple[int, int]]],
                labels: Optional[torch.Tensor],
                **kwargs,
                ) -> dict:

        hidden_states = self.encoder()

        output_dict = {
            "logits": classification_output.output_logits,
            "probabilities": classification_output.output_probs,
            "predictions": classification_output.output_predictions,
        }

        if classification_output.loss is not None:
            output_dict["loss"] = classification_output.loss

        return output_dict


    # todo: proper optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
