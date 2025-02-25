import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import joblib
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class EEGTransformer(nn.Module):
    def __init__(self, maxlen, num_features, num_classes, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EEGTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, maxlen, embed_dim))
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # Change shape for pooling
        x = self.global_avg_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=150):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch + 1}: Loss = {total_loss / total:.4f}, Val Acc = {val_acc:.4f}")
        
        
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return correct / total

def modal(data_list,label_list):
    X = np.array(data_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    joblib.dump(scaler, 'scaler.pkl')

    dataset = EEGDataset(X, y)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = EEGTransformer(maxlen=401, num_features=64, num_classes=8, embed_dim=32, num_heads=7, ff_dim=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer)
    test_acc = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")