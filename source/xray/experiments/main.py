import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from pathlib import Path
import time
import copy


def get_data():
    x_train = torch.load('data/x_train.pt')
    x_test = torch.load('data/x_test.pt')
    y_train = torch.load('data/y_train.pt')
    y_test = torch.load('data/y_test.pt')

    print('x_train.shape:', x_train.shape)
    print('y_train.shape:', y_train.shape)
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)

    datasets = {
        'train': data.TensorDataset(x_train, y_train),
        'val': data.TensorDataset(x_test, y_test)
    }

    dataloaders = {x: data.DataLoader(datasets[x], batch_size=64, shuffle=True)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(datasets[x])
                     for x in ['train', 'val']}

    return dataloaders, dataset_sizes


def get_model():
    model = torchvision.models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # n_in = model.fc.in_features

    class AddDim(nn.Module):
        def forward(self, x):
            return x.unsqueeze(1)

    class Flatten(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)

    class Scalar(nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.weight = Parameter(torch.rand(in_features))
        def forward(self, input):
            return self.weight * input

    # class PrintShape(nn.Module):
    #     def forward(self, x):
    #         print(x.shape)
    #         return x

    model.fc = nn.Sequential(
        Scalar(512),
        AddDim(),
        nn.MaxPool1d(32),
        Flatten(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )

    return model


def train(path_model, n_epochs=200):
    time_start = time.time()

    use_cuda = False
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print('device:', device)

    dataloaders, dataset_sizes = get_data()

    model = get_model()
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    best_model_state = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for i_epoch in range(n_epochs):
        print('Epoch {}/{}'.format(i_epoch, n_epochs - 1))

        for phase in ['train', 'val']:
            model.train() if (phase == 'train') else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((outputs > 0.5) == labels.byte())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - time_start
    print('Elapsed time {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), path_model)


def test(path_model):
    use_cuda = False
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print('device:', device)

    model = get_model()
    model.load_state_dict(torch.load(path_model))
    model.eval()
    model.to(device)

    # w = model.fc[4].weight
    # print(w)
    # a = torch.argsort(w, descending=True)
    # print(a[:20])
    # print(w[a[:20]])
    # return

    mode = 'test'
    x_test = torch.load('data/x_{}.pt'.format(mode))
    y_test = torch.load('data/y_{}.pt'.format(mode))

    for x, y in zip(x_test, y_test):
        input = x.unsqueeze(0).to(device)
        output = model(input)
        output = output.cpu().squeeze(0)
        print(y.item(), output.item())


def to01range(x):
    x = x - x.min()
    x = x / x.max()
    return x


if __name__ == '__main__':
    path_model = Path('data/model.pt')
    # train(path_model)
    test(path_model)
