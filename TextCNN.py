import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_classes=2, kernel_sizes=(8, 9, 10), channels=128):
        super(TextCNN, self).__init__()

        # define embedding table
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # define the convolutional layer (multiple kernels)
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=channels, kernel_size=k) for k in kernel_sizes])
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(len(kernel_sizes) * channels, num_classes)

    def forward(self, x):
        x_embedding = self.embedding(x)

        x_embedding = x_embedding.permute(0, 2, 1)
        conv_outputs = []

        for conv in self.conv_layers:
            conv_outputs.append(nn.functional.relu(conv(x_embedding)))

        pooled_outputs = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conv_outputs]

        x_embedding = torch.cat(pooled_outputs, dim=1)
        x_embedding = self.dropout(x_embedding)
        x_embedding = self.fc(x_embedding)

        return x_embedding