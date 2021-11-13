import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 dropout,
                 num_classes,
                 pad_idx
                 ):
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = num_classes
        self.pad_idx = pad_idx

        """
        
        """

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=self.pad_idx
        )

        self.rnn = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.linear_layer = nn.Linear(
            in_features=2 * self.hidden_size,
            out_features=self.num_classes
        )

    def forward(self, texts):

        """

        """

        embedded = self.embedding(texts)

        output, hidden_state = self.rnn(embedded)

        hidden_state = hidden_state.view(self.num_layers, 2, hidden_state.size()[1], self.hidden_size)

        concatenated_hidden_state = torch.cat((hidden_state[-1][0], hidden_state[-1][1]), 1)

        dropout = self.dropout_layer(concatenated_hidden_state)

        output = self.linear_layer(dropout)

        return output
