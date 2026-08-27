import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

area = [
    [50], [55], [60], [65], [70],
    [75], [80], [85], [90], [95],
    [100], [105], [110], [115], [120],
    [125], [130], [135], [140], [145],
    [150], [160], [170], [180], [200]
]

price = [
    [250], [275], [300], [325], [350],
    [375], [400], [425], [450], [475],
    [500], [525], [550], [575], [600],
    [625], [650], [675], [700], [725],
    [750], [800], [850], [900], [1000]
]


X = torch.tensor(area).to(torch.float32)
y = torch.tensor(price).to(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()


predictions = model(X_train)

loss_fn = nn.MSELoss()
loss = loss_fn(predictions,y_train)

loss.backward()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
print("before:", model.linear.weight)
print("before:", model.linear.bias)

optimizer.step()

print("after:", model.linear.weight)
print("after:", model.linear.bias)