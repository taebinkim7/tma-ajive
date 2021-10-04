import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.bn1(self.fc1(inputs)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.output(x)

        return x


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
        # self.fc1 = nn.Linear(grid_size * grid_size * 512, 1)
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
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.dropout(x)
        # x = self.relu(self.bn3(self.conv3(x)))
        # x = self.dropout(x)
        # x = self.relu(self.bn4(self.conv4(x)))
        # x = self.dropout(x)
        # x = self.pool(x)

        # x = self.relu(self.bn5(self.conv5(x)))
        # x = self.dropout(x)
        # x = self.relu(self.bn6(self.conv6(x)))
        # x = self.dropout(x)
        # x = self.relu(self.bn7(self.conv7(x)))
        # x = self.dropout(x)
        # x = self.relu(self.bn8(self.conv8(x)))
        # x = self.dropout(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        # x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.output(x)

        return x


class GetDataset(Dataset):
  def __init__(self, X, y=None, model_type='cnn'):
    self.X = torch.tensor(X, dtype=torch.float32)
    if model_type.lower() == 'cnn':
        self.X = self.X.permute(0, 3, 1, 2)
    self.y = torch.tensor(y, dtype=torch.float32)
    self.length = len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __len__(self):
    return self.length


def binary_acc(y_pred, y_test):
    y_pred_ = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_ == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def nn_classification(X, y, model_type='cnn', p_train=.8, seed=None, epochs=30,
                      batch_size=64, learning_rate=.001):
    # get training and test set
    n = len(y)
    perm_idx = np.random.RandomState(seed).permutation(np.arange(n))
    train_idx = perm_idx[:int(p_train * n)]
    test_idx = perm_idx[int(p_train * n):]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_dataset = GetDataset(X_train, y_train, model_type)
    test_dataset = GetDataset(X_test, y_test, model_type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # set model
    if model_type.lower() == 'cnn':
        model = CNN(6)
    elif model_type.lower() == 'mlp':
        model = MLP()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    model.train()
    for i in range(1, epochs + 1):
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

        print(f'Epoch {i + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} \
            | Train Acc: {epoch_acc / len(train_loader):.3f}')

        y_pred_list = []
        model.eval()
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                y_batch_pred = model(X_batch)
                y_batch_pred = torch.sigmoid(y_batch_pred)
                y_batch_pred = torch.round(y_batch_pred)
                y_pred_list.append(y_batch_pred.cpu().numpy())

        y_pred = np.array([a.squeeze().tolist() for a in y_pred_list])

        acc = np.round(100 * np.mean(y_test == y_pred), 1)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tp_rate = np.round(100 * tp / (tp + fn), 1)
        tn_rate = np.round(100 * tn / (tn + fp), 1)
        precision = np.round(100 * tp / (tp + fp), 1)

        print('Acc: {}, TP rate: {}, TN rate: {}, Precision: {}'\
            .format(acc, tp_rate, tn_rate, precision))

    return acc, tp_rate, tn_rate, precision
