import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super(LogisticRegression, self).__init__()

        # Embedding table
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim * 700, 1)

    def forward(self, x):
        x_embedding = self.embedding_table(x)
        x_embedding = x_embedding.view(32, 64 * 700)

        y_pred = torch.sigmoid(self.linear(x_embedding))
        return y_pred

