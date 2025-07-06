import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class GestureDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.sequences = []
        self.labels = []
        self.gesture_to_idx = {}
        self.preprocess_data()

    def preprocess_data(self):
        length = 120
        gestures = self.data['Gesture'].unique()
        self.gesture_to_idx = {gesture: idx for idx, gesture in enumerate(gestures)}

        for gesture in gestures:
            gesture_data = self.data[self.data['Gesture'] == gesture]
            for i in range(0, len(gesture_data), length):
                seq = gesture_data.iloc[i:i + length]
                if len(seq) == length:
                    sequence = seq[['Roll', 'Pitch', 'Thumb', 'Index', 'Middle', 'Ring', 'Little']].values
                    self.sequences.append(sequence)
                    self.labels.append(self.gesture_to_idx[gesture])

        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)

        scaler = StandardScaler()
        self.sequences = scaler.fit_transform(
            self.sequences.reshape(-1, self.sequences.shape[-1])
        ).reshape(self.sequences.shape)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)


class DFFN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DFFN, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv4 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.lstm = nn.LSTM(1024, 128, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        x_deep = torch.relu(self.conv4(x))
        x_deep = torch.relu(self.conv5(x_deep))
        x_deep = torch.relu(self.conv6(x_deep))

        x = torch.cat((x, x_deep), dim=1)
        x, _ = self.lstm(x.permute(0, 2, 1))
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


if __name__ == "__main__":
    dataset = GestureDataset("gesture_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.sequences, dataset.labels, test_size=0.2, random_state=42)

    train_data = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long))
    test_data = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFFN(input_dim=7, num_classes=12).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.permute(0, 2, 1).to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.permute(0, 2, 1).to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch + 1}/50], Test Accuracy: {correct / total:.4f}')