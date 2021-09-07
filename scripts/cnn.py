import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from joblib import load
from glob import glob
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tma_ajive.Paths import Paths


tensors = load(os.path.join(Paths().features_dir, 'core_tensors_er'))
labels = pd.read_csv(os.path.join(Paths().classification_dir,
                                  'core_labels_er.csv'), index_col=0)

intersection = list(set(tensors.keys()).intersection(set(labels.index)))
X = np.array([tensors[id] for id in intersection])
y = labels.loc[intersection]['er_label'].to_numpy().astype(int)

n = len(y)
perm_idx = np.random.RandomState(seed=111).permutation(np.arange(n))
train_idx, test_idx = perm_idx[:int(.8 * n)], perm_idx[int(.8 * n):]

train_X, test_X = X[train_idx], X[test_idx]
train_y, test_y = y[train_idx], y[test_idx]

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001

class dataset(Dataset):
  def __init__(self, X, y=None):
    self.X = torch.tensor(X, dtype=torch.float32)
    self.X = self.X.permute(0, 3, 1, 2)
    self.y = torch.tensor(y, dtype=torch.float32)
    self.length = len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __len__(self):
    return self.length

train_dataset = dataset(train_X, train_y)
test_dataset = dataset(test_X, test_y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

class CNN(nn.Module):
    def __init__(self, grid_size=6):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(grid_size * grid_size * 512 // 4, 2048)
        self.fc1 = nn.Linear(grid_size * grid_size * 512, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.output = nn.Linear(2048, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)

    def forward(self, inputs):
        x = self.relu(self.bn1(self.conv1(inputs)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        # x = self.pool(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        # x = self.relu(self.bn7(self.conv7(x)))
        # x = self.dropout(x)
        # x = self.relu(self.bn8(self.conv8(x)))
        # x = self.dropout(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.output(x)

        return x

model = CNN(6)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = np.array([a.squeeze().tolist() for a in y_pred_list])

    tn, fp, fn, tp = confusion_matrix(test_y, y_pred_list).ravel()
    tp_rate = np.round(tp / (tp + fn), 3)
    tn_rate = np.round(tn / (tn + fp), 3)
    acc = np.round(np.mean(test_y == y_pred_list), 3)
    print('Acc: {}, TP rate: {}, TN rate: {}'.format(acc, tp_rate, tn_rate))
