import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class GestureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class MFA_Block(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gap = nn.functional.adaptive_avg_pool1d(x, 1)
        alpha = self.sigmoid(self.conv2(nn.functional.relu(self.conv1(gap))))
        return alpha * x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv1d(input_dim, input_dim, kernel_size,
                                   groups=input_dim, padding=kernel_size // 2)
        self.pointwise = nn.Conv1d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = nn.functional.relu(self.depthwise(x))
        return self.pointwise(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        return self.encoder(x)


class GestureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class STFTnet(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, hidden_dim=64, num_layers=2):
        super().__init__()
        self.mfa = MFA_Block(input_dim, hidden_dim // 2)
        self.projection_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.depthwise_conv = DepthwiseSeparableConv(hidden_dim, hidden_dim)
        self.transformer = TransformerEncoderBlock(hidden_dim, num_heads, hidden_dim * 4, num_layers)
        self.classifier = GestureClassifier(hidden_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.mfa(x)
        x = self.projection_conv(x)
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        return self.classifier(x[:, -1, :])


if __name__ == "__main__":
    raw_data = pd.read_csv("gesture_data.csv")
    sequences = []
    labels = []
    scaler = MinMaxScaler(feature_range=(-1, 1))
    length = 120
    gestures = raw_data['Gesture'].unique()
    gesture_to_idx = {g: i for i, g in enumerate(gestures)}

    for gesture in gestures:
        gesture_data = raw_data[raw_data['Gesture'] == gesture]
        for i in range(0, len(gesture_data), length):
            seq = gesture_data.iloc[i:i + length]
            if len(seq) == length:
                seq_data = seq[['Roll', 'Pitch', 'Thumb', 'Index', 'Middle', 'Ring', 'Little']].values
                sequences.append(scaler.fit_transform(seq_data))
                labels.append(gesture_to_idx[gesture])

    sequences = np.array(sequences)
    labels = np.array(labels)
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    train_data = GestureDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_data = GestureDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = STFTnet(input_dim=7, num_classes=len(gestures))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

    for epoch in range(50):
        model.train()
        train_correct, train_total = 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}/50 | "
              f"Train Acc: {train_correct / train_total:.4f} | "
              f"Val Acc: {val_correct / val_total:.4f}")