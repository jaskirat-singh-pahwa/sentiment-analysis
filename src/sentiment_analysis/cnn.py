import torch
import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 output_channels,
                 filter_heights,
                 stride,
                 dropout,
                 num_classes,
                 pad_idx
                 ):
        super(CNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_channels = output_channels
        self.filter_heights = filter_heights
        self.stride = stride
        self.dropout = dropout
        self.num_classes = num_classes
        self.pad_idx = pad_idx

        """"
        1. First we will create an embedding layer to represent the words
            in our vocabulary
        
        2. Then, we will define multiple convolutional layers with a 
            filter (Kernel) size
        
        3. Then, we will create a dropout layer
        
        4. At last, we will create a linear layer
        
        """

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=self.pad_idx
        )

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.output_channels,
                    kernel_size=(w, self.embedding_size),
                    stride=self.stride
                ) for w in self.filter_heights
            ]
        )

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.linear_layer = nn.Linear(
            in_features=self.output_channels * len(self.filter_heights),
            out_features=self.num_classes
        )

    def forward(self, texts):

        """
        1. First, we will pass texts through our embedding layer to convert texts
            from word ids to word embeddings

        2. Since, out input to convolutional layer have 1 channel, therefore we will
            un-squeeze the embeddings

        3. Then, we will pass these text embeddings to our convolutional layers

        4. Then, we will apply dropout

        5. At last, we will pass our output from the dropout layer to our Linear layer

        """

        embedded = self.embedding(texts)
        embedded = embedded.unsqueeze(1)
        # print(embedded.shape)

        conv_2d = [
            f.relu(conv_layer(embedded)).squeeze(3)
            for conv_layer in self.conv_layers
        ]

        conv_2d_with_pooling = [
            f.max_pool1d(layer, layer.size(2)).squeeze(2)
            for layer in conv_2d
        ]

        concatenated_conv_2d = torch.cat(conv_2d_with_pooling, 1)

        dropout = self.dropout_layer(concatenated_conv_2d)

        output = self.linear_layer(dropout)

        return output
