from pathlib import Path
import pickle

from masks.dataset import get_dataloaders
from masks.segmentationmodel import SegmentationModel


def save_dataloaders():
    path_project = Path(__file__).parent.parent
    path_images = path_project / Path(r'data\xray_dataset\annotations_0\fake_f')
    path_annotations = path_project / Path(r'data\xray_dataset\annotations_0\annotations')
    path_dataloaders = path_project / Path(r'data\train_test\dataset_seg_batch4.pickle')

    dataloaders = get_dataloaders(path_images, path_annotations, batch_size=4)

    with open(path_dataloaders, 'wb') as file:
        pickle.dump(dataloaders, file)


def train():
    path_project = Path(__file__).parent.parent
    path_dataloaders = path_project / Path(r'data\train_test\dataset_seg_batch4.pickle')
    path_model = path_project / Path(r'data\models\model_unet.pt')

    with open(path_dataloaders, 'rb') as file:
        dataloaders = pickle.load(file)

    model = SegmentationModel()
    model.fit(dataloaders, n_epochs=2, path_save=path_model, use_cuda=False)


if __name__ == '__main__':
    # save_dataloaders()
    # train()
    pass
