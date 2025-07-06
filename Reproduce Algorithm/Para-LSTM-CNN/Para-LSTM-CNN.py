import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
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


class Para_LSTM_CN(nn.Module):
    def __init__(self, num_classes=12):
        super(Para_LSTM_CN, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 10, kernel_size=(1, 1), padding=0)
        self.conv1_2 = nn.Conv2d(1, 10, kernel_size=(3, 3), padding=1)
        self.conv1_3 = nn.Conv2d(1, 10, kernel_size=(5, 5), padding=2)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(30, 20, kernel_size=(1, 1), padding=0)
        self.conv2_2 = nn.Conv2d(30, 20, kernel_size=(3, 3), padding=1)
        self.conv2_3 = nn.Conv2d(30, 20, kernel_size=(5, 5), padding=2)

        self.conv3 = nn.Conv2d(60, 20, kernel_size=(3, 3), padding=1)

        self.lstm1 = nn.LSTM(input_size=140, hidden_size=60, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=20, batch_first=True)

        self.fc1 = nn.Linear(20, num_classes)

    def forward(self, x):
        x1_1 = torch.relu(self.conv1_1(x))
        x1_2 = torch.relu(self.conv1_2(x))
        x1_3 = torch.relu(self.conv1_3(x))
        x1 = torch.cat([x1_1, x1_2, x1_3], dim=1)

        x2_1 = torch.relu(self.conv2_1(x1))
        x2_2 = torch.relu(self.conv2_2(x1))
        x2_3 = torch.relu(self.conv2_3(x1))
        x2 = torch.cat([x2_1, x2_2, x2_3], dim=1)

        x3 = torch.relu(self.conv3(x2))

        x3 = x3.view(x3.size(0), 50, -1)
        x, _ = self.lstm1(x3)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]
        return self.fc1(x)


if __name__ == "__main__":
    file_path = "gesture_data.csv"
    dataset = GestureDataset(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.sequences, dataset.labels, test_size=0.2, random_state=42)

    train_data = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long))
    test_data = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Para_LSTM_CN(num_classes=12).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.unsqueeze(1))
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {correct / total:.4f}')