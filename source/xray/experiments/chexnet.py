import torch
from torch import nn
import torchvision
from torchvision import transforms
from pathlib import Path
import re
import cv2
from gradcam import GradCAM
from gradcam.utils import visualize_cam
import numpy as np
from tqdm import tqdm


class DenseNet121(nn.Module):
    def __init__(self, out_dim=14):
        super(DenseNet121, self).__init__()

        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        k = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(
            nn.Linear(k, out_dim),
            nn.Sigmoid()
        )

        # CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        #                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
        #                'Hernia']

    def forward(self, x):
        return self.densenet121(x)

    def load(self, path_model=Path('data/model.pth.tar')):
        checkpoint = torch.load(path_model, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        for key in list(state_dict.keys()):
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = new_key[7:]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

        self.load_state_dict(state_dict)


def get_model(path=None):
    model = DenseNet121()
    model.load()

    for param in model.parameters():
        param.requires_grad = False

    n_in = model.densenet121.classifier[0].in_features

    model.densenet121.classifier = nn.Sequential(
        nn.Linear(n_in, 1),
        nn.Sigmoid()
    )

    if path:
        model.load_state_dict(torch.load(path))

    return model


def main():
    path_model = Path('data/model_chexnet.pt')
    model = get_model(path_model)
    for param in model.parameters():
        param.requires_grad = True

    path_data = Path(r'C:\Downloads\Xray_samples_real_and_fake_age18-24\real\m')
    path_save_dir = Path('data/cam_result_real_chexnet')
    k1 = 160. / 512
    k2 = 470. / 512

    transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    gradcam = GradCAM(model, model.densenet121.features[-1])

    for path_image in tqdm(path_data.iterdir()):

        image = cv2.imread(str(path_image))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image[int(k1 * image.shape[0]):int(k2 * image.shape[0])]
        left = image[:, :image.shape[1] // 2]
        right = image[:, image.shape[1]//2:][:, ::-1]
        left = cv2.resize(left, (224, 224))
        right = cv2.resize(right, (224, 224))
        left_ = transforms.ToTensor()(left)
        right_ = transforms.ToTensor()(right)
        left = transform(left_)
        right = transform(right_)

        mask, pred_left = gradcam(left.unsqueeze(0))
        _, left = visualize_cam(mask, left_)
        left = left.permute(1, 2, 0).numpy()

        mask, pred_right = gradcam(right.unsqueeze(0))
        _, right = visualize_cam(mask, right_)
        right = right.permute(1, 2, 0).numpy()

        cam = (np.hstack((left, right[:, ::-1])) * 255.0).astype(int)

        path_save = path_save_dir / '{:.3f}_{:.3f}_{}.png'.format(pred_left.item(), pred_right.item(), path_image.stem)
        cv2.imwrite(str(path_save), cam)


if __name__ == '__main__':
    main()
