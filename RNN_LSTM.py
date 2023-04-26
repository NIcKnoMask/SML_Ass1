import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size=5000, embedding_dim=64, hidden_dim=128, layer_dim=2, num_classes=2):
        super(LSTM, self).__init__()

        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.layer_dim = layer_dim  # nums of rnn layers
        self.hidden_size = hidden_dim  # nums of rnn neural
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedding = self.embedding_table(x)

        r_out, _ = self.lstm(embedding, None)
        output = self.fc(r_out[:, -1, :])
        return output
