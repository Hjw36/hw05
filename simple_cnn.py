# simpleCNN.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 定义简单CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 输入: [batch, 1, 28, 28]
        x = self.pool(self.relu(self.conv1(x)))  # -> [batch, 32, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # -> [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 数据加载函数
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

# 3. 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    train_loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_loss_history.append(epoch_loss)
        print(f"第{epoch+1}个epoch: 准确率{epoch_acc:.2f}%, 损失{epoch_loss:.4f}")
    return train_loss_history

# 4. 测试函数
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total
    print(f"\n测试准确率: {accuracy:.2f}%")
    print(f"测试损失: {avg_loss:.4f}")
    return accuracy, avg_loss

# 5. 绘制损失曲线
def plot_loss(train_loss_history):
    plt.plot(train_loss_history, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# 主函数：把整个流程封装在这里
def main():
    # 超参数设置
    epochs = 5
    lr = 0.001

    # 加载数据
    train_loader, test_loader = load_data()

    # 初始化模型、损失函数、优化器
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    print("模型训练过程：")
    train_loss_history = train_model(model, train_loader, criterion, optimizer, epochs)

    # 测试模型
    test_acc, test_loss = test_model(model, test_loader, criterion)

    # 绘制损失曲线
    plot_loss(train_loss_history)

# 关键：加上这个判断，直接运行文件时才执行main()，导入时不执行
if __name__ == '__main__':
    main()