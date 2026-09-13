import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = load_breast_cancer()

X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_test, X_val, y_test, y_val = train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    random_state=42,
    stratify=y_test
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


class BreastCancerModel(nn.Module):

    def __init__(self):
        super(BreastCancerModel, self).__init__()

        self.fc1 = nn.Linear(30, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = BreastCancerModel()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 100

training_losses = []
validation_losses = []

best_val_loss = float("inf")


for epoch in range(epochs):

    model.train()

    train_loss = 0

    for X_batch, y_batch in train_dataloader:

        y_batch = y_batch.unsqueeze(1)

        optimizer.zero_grad()

        output = model(X_batch)

        loss = criterion(output, y_batch)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)


    model.eval()

    val_loss = 0

    with torch.inference_mode():

        for X_batch, y_batch in val_dataloader:

            y_batch = y_batch.unsqueeze(1)

            output = model(X_batch)

            loss = criterion(output, y_batch)

            val_loss += loss.item()

    val_loss /= len(val_dataloader)

    training_losses.append(train_loss)
    validation_losses.append(val_loss)


    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "X_test": X_test,
            "y_test": y_test
        }, "best_model.pth")


    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Training Loss: {train_loss:.6f}, "
        f"Validation Loss: {val_loss:.6f}"
    )


plt.figure(figsize=(10, 6))

plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.show()