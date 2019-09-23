import torch
import torch.nn as nn
import numpy as np

from utils import calc_embedding_dims
from params import FEATURE_SIZES, FEATURE_NUMBER
from data.data_splitter import Splitter
from data.data_generator import DataGenerator
from losses import DiceLoss


class RnnRelatedModel(nn.Module):
    def __init__(self,
                 input_size: int = 12,
                 hidden_size: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 output_size: int = 55,
                 loss_weight: torch.tensor = None,
                 embedding_scale: int = 2):
        super(RnnRelatedModel, self).__init__()

        # self.embeddings = [
        #     nn.Embedding(size, calc_embedding_dims(size, embedding_scale)) for size in FEATURE_SIZES
        # ]
        # total_embedding_size = sum([calc_embedding_dims(size, embedding_scale) for size in FEATURE_SIZES])
        self.rnn_layer = nn.LSTM(
            # input_size=total_embedding_size,
            input_size=12,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )
        self.dense_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.loss = nn.CrossEntropyLoss(weight=loss_weight)

    def forward(self, batch_input: np.ndarray, batch_label: np.ndarray, mode: str = "train"):
        """
        batch_input: (batch_size, max_len, input_size)
        batch_label: (batch_size, ) - **unfixed_length**
        """
        if isinstance(batch_input, np.ndarray):
            batch_input = torch.from_numpy(batch_input).float()
        if isinstance(batch_label, np.ndarray):
            batch_label = torch.from_numpy(batch_label).long()

        # embedding layer
        # embeddings = []
        # for i in range(FEATURE_NUMBER):
        #     per_embed = self.embeddings[i](batch_input[:, :, i])
        #     embeddings.append(per_embed)
        # embeddings = torch.cat(embeddings, dim=-1)
        embeddings = nn.functional.softmax(batch_input, dim=1)

        hidden, (last_hidden, last_cell) = self.rnn_layer(embeddings)
        last_hidden = torch.squeeze(last_hidden, dim=0)
        # (batch_size, hidden_size)
        dense_output = self.dense_layer(last_hidden)
        dropout_output = self.dropout(dense_output)
        # (batch_size, output_size)
        output = self.output_layer(dropout_output)

        if mode == "train":
            loss = self.loss(output, batch_label)
            return output, loss

        return output, None


if __name__ == '__main__':
    splitter = Splitter(
        label_list=[0, 2, 7, 17],
        is_slice_data=False,
        is_create_negative_sample=True
    )
    splitter.split()
    generator = DataGenerator(splitter, 10)

    model = RnnRelatedModel(
        input_size=12,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        output_size=splitter.get_label_size()
    )
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.9)
    for idx, data in enumerate(generator.generate(batch_size=20)):
        optimizer.zero_grad()
        preds, loss = model(data[0], data[1], "train")
        print(loss.data)
        loss.backward()
        optimizer.step()
