import torch
from torchvision import transforms
import cv2
import numpy as np
import requests


class GradCam:
    def __init__(self, model, use_cuda=True):
        self.model = model
        self.model.eval()

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = model.cuda()

        self.gradients = []
        self.target_activation = None

        labels_url = 'https://s3.amazonaws.com/mlpipes/pytorch-quick-start/labels.json'
        self.labels_imagenet = {int(key): value for (key, value) in requests.get(labels_url).json().items()}

    def forward_model(self, x):
        ''' self.target_activation = x '''
        raise NotImplementedError

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x, index=None):
        if self.use_cuda:
            x = x.cuda()

        output = self.forward_model(x).cpu()
        handle = self.target_activation.register_hook(self.save_gradient)

        if index is None:
            index = np.argmax(output.data.numpy())
        print(self.labels_imagenet[index])

        one_hot = np.zeros((1, output.size()[-1])).astype(np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.gradients[-1].cpu().data.numpy()

        handle.remove()
        self.gradients.clear()

        target = self.target_activation.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam = cam - np.min(cam)
        m = np.max(cam)
        if m != 0:
            cam = cam / m

        return cam


def cam_on_image(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def preprocess_image(image):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = image.unsqueeze(0)

    return image


def read_image(path_image, size):
    image = cv2.imread(str(path_image))
    image = cv2.resize(image, (size, size))
    image = np.float32(image) / 255.0
    return image


# Usage example.
# def main():
#
#     path_image = Path('../data/images/panda.jpg')
#     size = 224
#
#     image = read_image(path_image, size)
#     input = preprocess_image(image)
#
#     mask = GradCam(model)(input)
#
#     cam = cam_on_image(image, mask)
#
#     cv2.imwrite('../data/cams/cam.jpg', cam)
