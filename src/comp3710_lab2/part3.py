import numpy as np
import torch
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Part 3.1：沿用讲义 Part 2 的 LFW 数据和训练/测试划分。
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.images
Y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
h, w = X.shape[1:]

# 讲义中的 LFW 像素已经在 [0, 1]，无需再除以 255。
print("X_min:", X.min(), "X_max:", X.max())
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
)
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print("X_train shape:", X_train.shape)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 讲义未指定这些训练参数；这里使用 batch_size=32、20 个 epoch。
batch_size = 32
epochs = 20
train_data = TensorDataset(
    torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
)
test_data = TensorDataset(
    torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# 两层卷积均为 3x3、32 个滤波器，后接全连接分类。
# ReLU、2x2 最大池化和 128 维隐藏全连接层用于补齐讲义未给出的网络细节。
pooled_h = ((h - 2) // 2 - 2) // 2
pooled_w = ((w - 2) // 2 - 2) // 2
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * pooled_h * pooled_w, 128),
    nn.ReLU(),
    nn.Linear(128, n_classes),
).to(device)
print(model)

# 按讲义使用 Adam 和类别交叉熵；CrossEntropyLoss 直接接收 logits。
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {total_loss / len(train_data):.4f}, "
        f"Training accuracy: {total_correct / len(train_data):.4f}"
    )

# 训练完成后在与 Part 2 相同的测试集上评价。
model.eval()
predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions.extend(outputs.argmax(dim=1).cpu().tolist())

predictions = np.array(predictions)
correct = predictions == y_test
total_test = len(y_test)
print("Total Testing", total_test)
print("Total Correct", np.sum(correct))
print("Accuracy", np.sum(correct) / total_test)
print(classification_report(y_test, predictions, target_names=target_names))
