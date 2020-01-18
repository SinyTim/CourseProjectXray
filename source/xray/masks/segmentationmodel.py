import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import copy
import cv2
import numpy as np

from masks.unet.unet_model import UNet


class SegmentationModel:

    def __init__(self, path=None):

        self.model = UNet(n_channels=1, n_classes=1)

        if path:
            self.load(path)

    def fit(self, dataloaders, n_epochs, use_cuda=True, path_save=None):
        """
        :param dataloaders: dict with keys 'train' and 'val'
        """

        device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        print('device:', device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-6)

        best_model_state = copy.deepcopy(self.model.state_dict())
        best_loss = 1e9

        for i_epoch in range(n_epochs):
            print('Epoch {}/{}'.format(i_epoch, n_epochs - 1))

            for phase in ['train', 'val']:
                self.model.train() if (phase == 'train') else self.model.eval()

                running_loss = 0.0

                for x, y in tqdm(dataloaders[phase]):
                    x = x.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        out = self.model(x)
                        loss = criterion(out, y)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() / x.shape[0]

                epoch_loss = running_loss / len(dataloaders[phase])
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())

        print('Best val loss: {:4f}'.format(best_loss))
        self.model.load_state_dict(best_model_state)

        if path_save:
            self.save(path_save)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def get_mask(self, image, use_cuda=True):
        """
        :param image: numpy 2d array [0..255]
        """

        device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        x = torch.from_numpy(image)
        x = x.unsqueeze(0).unsqueeze(0)
        x = x.to(device)

        mask = self.model(x)
        mask = (mask[0][0] > 0.5).detach().cpu().numpy()

        return mask

