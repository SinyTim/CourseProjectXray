import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from experiments.grad_cam import preprocess_image
from experiments.grad_cam import cam_on_image


class ModelOutputs:

    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = []
        self.target_layers = target_layers

    def get_gradients(self):
        return self.gradients

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):

        target_activations = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if name == 'fc':
                break
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                target_activations += [x]

        x = x.reshape(x.size(0), -1)
        x = self.model.fc(x)

        return target_activations, x


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda=False):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=0):
        if self.cuda:
            input = input.cuda()
        features, output = self.extractor(input)
        # print(output)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        if self.cuda:
            one_hot = one_hot.cuda()
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        self.extractor.gradients.clear()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = cam_init = np.maximum(cam, 0)
        # cam = cam_init = (cam > 0.017).astype(np.float)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        m = np.max(cam)
        if m != 0:
            cam = cam / m
        return cam, output[0][index].item(), cam_init


def main():

    path_model = Path('data/model.pt')
    model = get_model()
    model.load_state_dict(torch.load(path_model))
    for param in model.parameters():
        param.requires_grad = True
    model.eval()

    path_data = Path(r'C:\Downloads\xray_dataset\Xray_samples_real_and_fake_age18-24\fake\f')
    path_save_dir = Path('data/cam_result_fake')
    # path_save_dir = Path('data/real_selection')
    # path_save_dir_neg = Path('data/real_unselected')
    # path_save_f = Path(r'data\temp\f')
    # path_save_m = Path(r'data\temp\m')

    k1 = 250. / 512
    k2 = 470. / 512

    # class Identity(torch.nn.Module):
    #     def __init__(self):
    #         super(Identity, self).__init__()
    #     def forward(self, input):
    #         return input
    #
    # model.fc = Identity()

    coss = []
    for path_image in tqdm(path_data.iterdir()):

        image = cv2.imread(str(path_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[int(k1 * image.shape[0]):int(k2 * image.shape[0])]
        left = image[:, 10:image.shape[1] // 2]
        right = image[:, image.shape[1]//2:-10][:, ::-1]
        left = cv2.resize(left, (224, 224))
        right = cv2.resize(right, (224, 224))
        left = np.float32(left) / 255.0
        right = np.float32(right) / 255.0
        input0 = preprocess_image(left)
        input1 = preprocess_image(right)

        # pred0 = model(input0)[0]
        # pred1 = model(input1)[0]
        # print((pred0 - pred1).pow(2).sum())

        # for pred, img in zip([pred0, pred1], [left, right]):
        #     suffix = '{:.3f}_{}.png'.format(pred, path_image.stem)
        #     cv2.imwrite(str(path_save_f / suffix if pred > 0.5 else path_save_m / suffix), np.uint8(255 * img))

        mask0, pred0, cam_init0 = GradCam(model, target_layer_names=['layer4'])(input0)
        mask1, pred1, cam_init1 = GradCam(model, target_layer_names=['layer4'])(input1)

        cos = (cam_init0 * cam_init1).sum() / np.linalg.norm(cam_init0) / np.linalg.norm(cam_init1)
        coss += [cos]

        # if pred0 < 0.5 or pred1 < 0.5 or abs(pred1 - pred0) > 0.2 or cos < 0.8:
        #     path_save = path_save_dir_neg / '{:.3f}_{:.3f}_{:.3f}_{}.png'.format(cos, pred0, pred1, path_image.stem)
        # else:
        #     path_save = path_save_dir / '{:.3f}_{:.3f}_{:.3f}_{}.png'.format(cos, pred0, pred1, path_image.stem)

        # cv2.imwrite(str(path_save), image)
        # continue

        cam0 = cam_on_image(left, mask0)
        cam1 = cam_on_image(right, mask1)

        cam = np.hstack((cam0, cam1[:, ::-1]))

        path_save = path_save_dir / '{:.3f}_{:.3f}_{:.3f}_{}.png'.format(cos, pred0, pred1, path_image.stem)
        cv2.imwrite(str(path_save), cam)

    # coss = np.array(coss)
    # print('median: {}, mean: {}, std: {}, max: {}'.format(np.median(coss), coss.mean(), coss.std(), coss.max()))
    # real 0.81910217 0.76504177 0.17574058 0.977247
    # fake 0.6325844 0.6146305 0.20141561 0.9649602


if __name__ == '__main__':
    main()
