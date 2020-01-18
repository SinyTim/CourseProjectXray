import torch
from torch.utils import data
import cv2
import numpy as np


class SegmentationDataset(data.Dataset):

    def __init__(self, path_images, path_annotations):
        """
        images: len x 1x256x256, [0.0 ... 1.0]
        annotations: len x 1x256x256, 0.0 or 1.0
        """

        images = []
        annotations = []

        for path_image in path_images.iterdir():
            path_annotation = path_annotations / path_image.name

            image = cv2.imread(str(path_image), cv2.IMREAD_GRAYSCALE)

            annotation = cv2.imread(str(path_annotation))
            annotation = annotation[..., 2] > 250

            images.append(image)
            annotations.append(annotation)

        images = np.array(images).astype(np.float32) / 255.0
        annotations = np.array(annotations).astype(np.float32)

        self.images = torch.from_numpy(images).unsqueeze(1)
        self.annotations = torch.from_numpy(annotations).unsqueeze(1)

    def __getitem__(self, index):
        return self.images[index], self.annotations[index]

    def __len__(self):
        return len(self.images)


def get_dataloaders(path_images, path_annotations, test_split=0.2, batch_size=16):

    dataset = SegmentationDataset(path_images, path_annotations)
    print('Dataset len: {}, x shape {}, y shape {}'.format(len(dataset), dataset[0][0].shape, dataset[0][1].shape))

    test_len = int(len(dataset) * test_split)
    train_len = len(dataset) - test_len
    dataset_train, dataset_test = data.random_split(dataset, (train_len, test_len))

    datasets = {
        'train': dataset_train,
        'val': dataset_test
    }

    dataloaders = {x: data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}

    return dataloaders
