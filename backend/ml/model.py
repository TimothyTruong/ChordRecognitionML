import torch
import torch.nn as nn
import torch.nn.functional as F

class ChordModelV1(nn.Module):
    def __init__(self, num_layers=2, hidden_size=64, num_classes=25, num_chroma_bins=12):
        super(ChordModelV1, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_chroma_bins, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for CNN
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.permute(0, 2, 1)  # Reshape for LSTM
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])  # Take last hidden state

        return output

class ChordModelV2(nn.Module):
    def __init__(self, num_layers=2, hidden_size=128, num_classes=25, num_chroma_bins=12):
        super(ChordModelV2, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_chroma_bins, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)  # Add batch normalization
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # BiLSTM → hidden_size * 2

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for CNN (batch, chroma_bins, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.permute(0, 2, 1)  # Reshape for LSTM (batch, seq_len, features)
        lstm_out, (hidden, _) = self.lstm(x)

        # Use mean of all LSTM outputs instead of just last hidden state
        output = self.fc(torch.mean(lstm_out, dim=1))

        return output