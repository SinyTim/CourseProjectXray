import cv2
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt


def read_directory(path):
    k1 = 250. / 512
    k2 = 470. / 512
    images = []

    for path_image in path.iterdir():
        image = cv2.imread(str(path_image))  # , cv2.IMREAD_GRAYSCALE
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[int(k1*image.shape[0]):int(k2*image.shape[0])]
        left = image[:, 10:image.shape[1]//2]
        right = image[:, image.shape[1]//2:-10][:, ::-1]
        left = cv2.resize(left, (224, 224))
        right = cv2.resize(right, (224, 224))
        images += [left, right]

    return np.array(images).astype(np.float32) / 255.0


def preprocess_data():
    path_data = Path(r'C:\Downloads\xray_dataset\real')
    path_real_female = path_data / 'f'
    path_real_male = path_data / 'm'

    images_real_female = read_directory(path_real_female)
    images_real_male = read_directory(path_real_male)

    labels_real_female = np.array([[1]] * len(images_real_female))
    labels_real_male = np.array([[0]] * len(images_real_male))

    images_real = np.concatenate((images_real_female, images_real_male))
    labels_real = np.concatenate((labels_real_female, labels_real_male))

    x_train, x_test, y_train, y_test = train_test_split(images_real, labels_real, test_size=0.2, stratify=labels_real)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    x_train = torch.stack([transform(x) for x in x_train])
    x_test = torch.stack([transform(x) for x in x_test])

    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    torch.save(x_train, 'data/x_train.pt')
    torch.save(x_test, 'data/x_test.pt')
    torch.save(y_train, 'data/y_train.pt')
    torch.save(y_test, 'data/y_test.pt')


def check():
    x = torch.load('data/x_test.pt')
    x = x[199]
    x = x.numpy()
    x = x.transpose(1, 2, 0)
    x = x - x.min()
    x = x / x.max()
    x = (x * 255.0).astype(np.int16)
    plt.imshow(x)
    plt.show()


if __name__ == '__main__':
    # preprocess_data()
    # check()
    pass
