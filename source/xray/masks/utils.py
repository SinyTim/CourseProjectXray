import cv2
import numpy as np
from pathlib import Path
import pickle

from masks.segmentationmodel import SegmentationModel


def get_masks(path_images, path_save_masks=None, path_save_mask_images=None):
    """
    :param path_images: directory
    :param path_save_masks: .pickle
    :param path_save_mask_images: directory
    :return:
    """

    if path_save_mask_images:
        path_save_mask_images.mkdir(exist_ok=True)

    path_model = Path(__file__).parent.parent / Path('data/models/model_unet_0.pt')
    model = SegmentationModel(path_model)

    masks = dict()
    n = len(list(path_images.iterdir()))

    for i, path_image in enumerate(path_images.iterdir()):
        print('{}/{}'.format(i, n))

        image = cv2.imread(str(path_image), cv2.IMREAD_GRAYSCALE)
        mask = model.get_mask(image)
        masks[path_image.name] = mask

        if path_save_mask_images:
            image = put_mask_on_image(mask, image)
            path_save = path_save_mask_images / 'mask_{}'.format(path_image.name)
            cv2.imwrite(str(path_save), image)

    if path_save_masks:
        with open(path_save_masks, 'wb') as file:
            pickle.dump(masks, file)


def put_mask_on_image(mask, image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_mask = image.copy()
    image_mask[..., 2][mask.astype(bool)] = 255
    image = np.concatenate((image, image_mask), axis=1)
    return image
