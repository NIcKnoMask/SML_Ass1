import torch
import torch.nn as nn


# Try a Gate Convolutional Neural network
class GateCNN(nn.Module):
    def __init__(self, vocab_size=5000, embedding_dim=32, n_class=2):
        super(GateCNN, self).__init__()

        # Embedding table
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)

        # convolutional layer
        self.conv_A_1 = nn.Conv1d(embedding_dim, 32, 10, stride=4)
        self.conv_B_1 = nn.Conv1d(embedding_dim, 32, 10, stride=4)

        self.conv_A_2 = nn.Conv1d(32, 64, 10, stride=4)
        self.conv_B_2 = nn.Conv1d(32, 64, 10, stride=4)

        # full connect layer
        self.logist1 = nn.Linear(32, 128)
        self.logist2 = nn.Linear(128, n_class)

    def forward(self, x):
        # embedding the input
        x_embedding = self.embedding_table(x)

        # 1dCNN
        x_embedding = x_embedding.transpose(1, 2)
        A = self.conv_A_1(x_embedding)
        B = self.conv_B_1(x_embedding)
        H = A * torch.sigmoid(B)

        A = self.conv_A_1(H)
        B = self.conv_B_1(H)
        H = A * torch.sigmoid(B)

        # pooling
        pool_output = torch.mean(H, dim=-1)
        linear1_output = self.logist1(pool_output)
        logist = self.logist2(linear1_output)

        return logist

