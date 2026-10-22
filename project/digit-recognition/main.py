
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report



digits = load_digits()  
x = digits.data # type: ignore
y = digits.target # type: ignore

X_train , X_test , y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2,stratify=y)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train,y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

batch_X, batch_y = next(iter(train_loader))

class DigitClassifier(nn.Module):
    def __init__(self) :
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

model = DigitClassifier()

output = model(batch_X)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 30

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")
model.eval()

with torch.no_grad():
    test_outputs = model(X_test)

    predictions = test_outputs.argmax(dim=1)

    # TODO: محاسبه‌ی accuracy
    accuracy = (predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item()*100:.2f}%")

    cm = confusion_matrix(y_test.numpy(), predictions.numpy())
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(y_test.numpy(), predictions.numpy()))

torch.save(model.state_dict(), "digit_classifier.pth")

loaded_model = DigitClassifier()

# TODO: بارگذاری وزن‌های ذخیره‌شده
loaded_model.load_state_dict(torch.load("digit_classifier.pth"))

loaded_model.eval()

with torch.no_grad():
    sample = X_test[0].unsqueeze(0)
    output = loaded_model(sample)
    prediction = output.argmax(dim=1)

print("Predicted digit:", prediction.item())
print("Actual digit:", y_test[0].item())
model.eval()
loaded_model.eval()

with torch.no_grad():
    original_predictions = model(X_test).argmax(dim=1)
    loaded_predictions = loaded_model(X_test).argmax(dim=1)

print(
    "Predictions match:",
    torch.equal(original_predictions, loaded_predictions)
)