import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


import torch.onnx
import argparse
from dataset import CustomDataset

class ResudualBlock_1(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResudualBlock_1, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_filters, out_filters, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv1d(in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                     padding_mode='replicate')
        self.conv3 = torch.nn.Conv1d(out_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                     padding_mode='replicate')
        self.conv4 = torch.nn.Conv1d(out_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                     padding_mode='replicate')
        self.pool = torch.nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        s = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = x + s
        x = F.relu(x)
        x = self.pool(x)
        return x

class ResudualBlock_2(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResudualBlock_2, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_filters, out_filters, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv1d(in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                     padding_mode='replicate')
        self.conv3 = torch.nn.Conv1d(out_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                     padding_mode='replicate')
        self.conv4 = torch.nn.Conv1d(out_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                     padding_mode='replicate')
        self.conv5 = torch.nn.Conv1d(out_filters, out_filters, kernel_size=3, stride=1, padding=1,
                                     padding_mode='replicate')
        self.pool = torch.nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        s = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = x + s
        x = F.relu(x)
        x = self.pool(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.res1 = ResudualBlock_1(1, 16)
        self.res2 = ResudualBlock_1(16, 32)
        self.res3 = ResudualBlock_2(32, 64)
        self.res4 = ResudualBlock_2(64, 128)
        self.res5 = ResudualBlock_2(128, 128)
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=3, stride=3)
        # self.fc1 = torch.nn.Linear(10624, 128)
        # self.fc3 = torch.nn.Linear(128, 72)


        self.fc1 = torch.nn.Linear(10624, 400)
        self.fc3 = torch.nn.Linear(400, 72)


    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.training:
            x = self.drop(x)
            x = self.fc3(x)
            x = F.log_softmax(x, dim=1)
        return x



def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    loss_out = 0

    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)



        optimizer.zero_grad()
        loss.backward()
        print([x.grad for x in model.parameters()])
        optimizer.step()
        loss_out += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_dataset = r'E:\train\\'
    dataset = CustomDataset(path_dataset)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    testloader = DataLoader(test_set, shuffle=True, batch_size=32, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=32, pin_memory=True)
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    for t in range(50):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, net, criterion, optimizer, device)
        test_loop(testloader, net, criterion, device)
        torch.save(net.state_dict(), r'E:\Programs\PycharmProjects\NIRS_2\ResNet_weights\model_weights' + str(t + 1) + '.pth')


if __name__ == '__main__':
    main()