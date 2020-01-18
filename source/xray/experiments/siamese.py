import torch
from torch import nn
from torch import optim
from torch.utils import data
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import random
import time
import pickle
from tqdm import tqdm


class SiameseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature_model = torchvision.models.resnet18(pretrained=True)

        for param in self.feature_model.parameters():
            param.requires_grad = False
        # for param in self.feature_model.layer4:
        #     param.requires_grad = True

        self.features_dim = self.feature_model.fc.in_features

        self.feature_model.fc = nn.Identity()
        self.feature_model.layer2 = nn.Identity()
        self.feature_model.layer3 = nn.Identity()
        self.feature_model.layer4 = nn.Identity()

        self.fc = nn.Sequential(
            # nn.Dropout(0.1),
            # nn.Linear(self.features_dim, 1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = self.feature_model(x)
        y = self.feature_model(y)
        # z = torch.cat((x, y), dim=1)
        z = x * y
        z = self.fc(z)
        return z


class SiameseDataset(data.Dataset):

    def __init__(self, path_directory):
        super().__init__()
        size = 224
        self.images_left = []
        self.images_right = []
        k1 = 160. / 512
        k2 = 470. / 512

        for path_image in path_directory.iterdir():
            image = cv2.imread(str(path_image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[int(k1*image.shape[0]):int(k2*image.shape[0])]
            image_left = image[:, :image.shape[1] // 2 - 10]
            image_right = image[:, image.shape[1] // 2 + 10:][:, ::-1]
            image_left = cv2.resize(image_left, (size, size))
            image_right = cv2.resize(image_right, (size, size))
            self.images_left.append(image_left)
            self.images_right.append(image_right)

        self.images_left = np.array(self.images_left).astype(np.float32) / 255.0
        self.images_right = np.array(self.images_right).astype(np.float32) / 255.0

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.images_left = torch.stack([transform(x) for x in self.images_left])
        self.images_right = torch.stack([transform(x) for x in self.images_right])

    def __getitem__(self, i):
        index = i // 2
        is_same = bool(i % 2)

        left = self.images_left[index]

        if is_same:
            right = self.images_right[index]
        else:
            indexes = [i for i in range(self.images_right.shape[0]) if i != index]
            right_index = random.choice(indexes)
            right = self.images_right[right_index]

        return left, right, torch.tensor([is_same]).float()

    def __len__(self):
        return 2 * self.images_left.shape[0]


def train():
    path_data = Path('data/dataset.pickle')
    # path_images = Path(r'C:\Downloads\Xray_samples_real_and_fake_age18-24\real\f')
    # dataset = SiameseDataset(path_images)
    # with open(path_data, 'wb') as file:
    #     pickle.dump(dataset, file)

    with open(path_data, 'rb') as file:
        dataset = pickle.load(file)
    n_train = int(0.8 * len(dataset))
    dataset_train, dataset_test = data.random_split(dataset, [n_train, len(dataset) - n_train])
    print(1.0 / (sum([l.item() for x, y, l in dataset_train]) / sum([l.item() for x, y, l in dataset_test]) + 1.0))

    dataloader_train = data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    dataloader_test = data.DataLoader(dataset_test, batch_size=64, shuffle=True)
    dataloaders = {'train': dataloader_train, 'val': dataloader_test}
    datasets_size = {'train': len(dataset_train), 'val': len(dataset_test)}

    path_model = Path('data/model_siamese.pt')
    n_epochs = 300

    time_start = time.time()

    use_cuda = True
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print('device:', device)

    model = SiameseNetwork()
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

    for i_epoch in range(n_epochs):
        print('Epoch {}/{}'.format(i_epoch, n_epochs - 1))

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for left, right, is_same in dataloaders[phase]:
                left = left.to(device)
                right = right.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(left, right).cpu()
                    loss = criterion(output, is_same)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum((output > 0.5) == is_same.byte())

            epoch_loss = running_loss / datasets_size[phase] * dataloaders[phase].batch_size
            epoch_acc = running_corrects.double() / datasets_size[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - time_start
    print('Elapsed time {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(model.state_dict(), path_model)


def test():
    use_cuda = False
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print('device:', device)

    path_model = Path('data/model_siamese.pt')
    model = SiameseNetwork()
    # model.load_state_dict(torch.load(path_model))
    model.eval()
    model.to(device)

    path_data = Path('data/dataset.pickle')
    with open(path_data, 'rb') as file:
        dataset = pickle.load(file)
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=False)

    # import matplotlib.pyplot as plt
    # for i in range(400, 405):
    #     x, y, label = dataset[i]
    #     print(label)
    #     x = torch.cat((x, y.flip(dims=(2,))), dim=2)
    #     plt.imshow(x.permute(1, 2, 0))
    #     plt.show()
    # return

    acc = 0

    for left, right, is_same in tqdm(dataloader):
        left = left.to(device)
        right = right.to(device)
        output = model(left, right).cpu()
        acc += torch.sum((output > 0.5) == is_same.byte())

    print(acc.item())
    print(len(dataset))
    print(acc.float().item() / len(dataset))


if __name__ == '__main__':
    train()
    # test()
