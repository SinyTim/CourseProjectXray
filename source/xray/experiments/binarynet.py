import torch
from torch import nn
from torch import optim
from torch.utils import data
import cv2
from pathlib import Path
import numpy as np
import pickle


class BinaryNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),

            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),

            Flatten(),
            # nn.Dropout(p=0.3),
            nn.Linear(3136, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    return image


def read_dir(path):
    images = [read_image(path_image) for path_image in path.iterdir()]
    names = [path_image.name for path_image in path.iterdir()]
    return torch.stack(images), names


def get_data():
    path = Path(r'C:\Downloads\Xray_samples_real_and_fake_age18-24\fake\f')

    images, names = read_dir(path)

    path_labels = Path('data/labels.pickle')
    with open(path_labels, 'rb') as file:
        labels = pickle.load(file)
    labels = [labels[name] for name in names]
    labels = torch.tensor(labels).float().unsqueeze(1)

    dataset = data.TensorDataset(images, labels)
    loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    return loader


def main():
    device = torch.device('cuda')
    net = BinaryNet()
    net.to(device)

    loader = get_data()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(30):
        running_loss = 0.0
        running_acc = 0.0
        for x, label in loader:
            x = x.to(device)
            optimizer.zero_grad()
            output = net(x).cpu()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += float(((output > 0.5) == label.byte()).sum().item())
        print('{} loss {:.3f}, acc {:.3f}'.format(epoch, running_loss / 290, running_acc / 290))


if __name__ == '__main__':
    main()
