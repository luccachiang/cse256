import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# 定义数据集类
class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split("\t", 1)
                self.labels.append(int(label))
                self.data.append(text.lower())  # 转为小写

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 定义 Deep Average Network 模型
class DeepAverageNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DeepAverageNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)  # x: (batch_size, seq_length)
        x = x.mean(dim=1)      # 计算平均
        x = torch.relu(self.fc1(x))  # 隐藏层
        x = self.fc2(x)        # 输出层
        return x

# 数据预处理和训练
def train_model(device, vocab_size, embedding_dim=50, hidden_dim=10, batch_size=32, epochs=10):
    # dataset = TextDataset(file_path)
    # train_texts, val_texts, train_labels, val_labels = train_test_split(dataset.data, dataset.labels, test_size=0.2)
    trainset = TextDataset("data/train.txt")
    train_texts = trainset.data
    train_labels = trainset.labels
    valset = TextDataset("data/dev.txt")
    val_texts = valset.data
    val_labels = valset.labels

    # 创建词汇表
    word_to_idx = {word: idx for idx, word in enumerate(set(' '.join(train_texts + val_texts).split()))}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # 将文本转换为索引
    train_data = [[word_to_idx[word] for word in text.split()] for text in train_texts]
    val_data = [[word_to_idx[word] for word in text.split()] for text in val_texts]

    # 创建 DataLoader
    train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(val_data, val_labels)), batch_size=batch_size)

    model = DeepAverageNetwork(vocab_size=len(word_to_idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练模型
    for epoch in range(epochs):
        model.train()
        for texts, labels in train_loader:
            texts = torch.nn.utils.rnn.pad_sequence([torch.tensor(text) for text in texts], batch_first=True).to(device)
            labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # 验证模型
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts = torch.nn.utils.rnn.pad_sequence([torch.tensor(text) for text in texts], batch_first=True)
            labels = torch.tensor(labels)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# 使用示例
if __name__ == "__main__":
    train_model(device='cuda:4', vocab_size=1000)  # 请替换为你自己的数据集路径