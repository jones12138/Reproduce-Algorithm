import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class GestureDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.sequences = []
        self.labels = []
        self.preprocess_data()

    def preprocess_data(self):
        length = 120
        gestures = self.data['Gesture'].unique()
        gesture_to_idx = {gesture: idx for idx, gesture in enumerate(gestures)}
        for gesture in gestures:
            gesture_data = self.data[self.data['Gesture'] == gesture]
            for i in range(0, len(gesture_data), length):
                seq = gesture_data.iloc[i:i + length]
                if len(seq) == length:
                    sequence = seq[['Roll', 'Pitch', 'Thumb', 'Index', 'Middle', 'Ring', 'Little']].values
                    self.sequences.append(sequence)
                    self.labels.append(gesture_to_idx[gesture])
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class GestureRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(GestureRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height, channels)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    file_path = "gesture_data.csv"
    dataset = GestureDataset(file_path)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureRecognitionModel(num_classes=12).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.float().to(device), labels.long().to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch + 1}/50], Test Accuracy: {correct / total:.4f}')