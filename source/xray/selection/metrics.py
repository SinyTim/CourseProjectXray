import numpy as np
import cv2
import pickle


def get_metrics(path_masks, path_metrics):

    with open(path_masks, 'rb') as file:
        masks = pickle.load(file)

    metrics_data = dict()

    kernel = np.ones((5, 5), np.uint8)

    for name, mask in masks.items():
        mask_d = cv2.dilate(mask, kernel, iterations=1)

        left, right = bisect(mask)
        left_d, right_d = bisect(mask_d)

        metrics_data[name] = [
            intersection_over_union(left, right),
            cos(left, right),
            area_ratio(left, right),

            intersection_over_union(left_d, right_d),
            cos(left_d, right_d),
            area_ratio(left_d, right_d),
        ]

    with open(path_metrics, 'wb') as file:
        pickle.dump(metrics_data, file)


def bisect(mask):
    width = mask.shape[1]
    mask_left = mask[:, :width // 2]
    mask_right = mask[:, width // 2:][:, ::-1]
    return mask_left, mask_right


def intersection_over_union(x, y):
    if x.sum() + y.sum() == 0:
        return 1.0
    return (x * y).sum() / (x.sum() + y.sum() - (x * y).sum())


def cos(x, y):
    if x.sum() == 0 and y.sum() == 0:
        return 1.0
    elif x.sum() == 0 or y.sum() == 0:
        return 0.0
    return (x * y).sum() / np.sqrt(x.sum() * y.sum())


def area_ratio(x, y):
    if x.sum() + y.sum() == 0:
        return 0.0
    return (x.sum() / (x.sum() + y.sum()) - 0.5) ** 2
