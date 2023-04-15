import torch
import torch.nn as nn
from transformers import BertModel


EMBEDDING_SIZE = 768


class BertEncoder(nn.Module):
    def __init__(
            self,
            dropout,
    ):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

    def forward(self, input_ids, attention_mask):

        embeddings, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        return embeddings
