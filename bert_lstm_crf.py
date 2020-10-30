from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torchcrf import CRF
from transformers import (BertForTokenClassification)

from punc_dataset import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PuncBERTLstmCrfModel(BertForTokenClassification):
    def __init__(self, config):
        super(PuncBERTLstmCrfModel, self).__init__(config=config)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, device='cuda'):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        sequence_output, _ = self.lstm(sequence_output)

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]

        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=attention_mask_label.type(torch.uint8))
            return -1.0 * log_likelihood
        else:
            sequence_tags = self.crf.decode(logits)
            return sequence_tags
