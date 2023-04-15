import torch
import torch.nn as nn
from transformers import BertModel
from .convolution import ConvolutionLayer

EMBEDDING_SIZE = 768


class BertEncoder(nn.Module):
    def __init__(
            self,
            dropout,
            output_embedding_len
    ):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.conv_layer = ConvolutionLayer(
            embedding_len=EMBEDDING_SIZE,
            output_embedding_len=output_embedding_len * 2
        )

        self.fc = nn.Linear(output_embedding_len * 2, output_embedding_len)

    def forward(self, input_ids, attention_mask):

        embeddings, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        embeddings = self.conv_layer(embeddings)
        embeddings = self.fc(embeddings)

        return embeddings
