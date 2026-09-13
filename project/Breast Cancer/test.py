import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


path = os.path.join(
    os.path.dirname(__file__),
    "best_model.pth"
)

checkpoint = torch.load(
    path,
    weights_only=False
)

X_test = checkpoint["X_test"]
y_test = checkpoint["y_test"]


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
)


test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

test_dataloader = DataLoader(
    test_dataset,
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

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


criterion = nn.BCEWithLogitsLoss()

test_loss = 0

all_predictions = []
all_labels = []


with torch.inference_mode():

    for X_batch, y_batch in test_dataloader:

        y_batch = y_batch.unsqueeze(1)

        output = model(X_batch)

        probability = torch.sigmoid(output)

        prediction = (probability >= 0.5).float()

        loss = criterion(output, y_batch)

        test_loss += loss.item()

        all_predictions.extend(
            prediction.squeeze().tolist()
        )

        all_labels.extend(
            y_batch.squeeze().tolist()
        )


test_loss /= len(test_dataloader)

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)


print(f"Test Loss: {test_loss:.6f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()


print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Benign", "Malignant"]
    )
)