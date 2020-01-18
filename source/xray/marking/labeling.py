import cv2
import pickle
from pathlib import Path


def labeling():
    path_dir = Path(r'data\xray_dataset\annotations_1\18_24f')
    path_save = Path(r'data\xray_dataset\annotations_1\labels_good_bad.pickle')

    map_key_label = {
        ord('z'): True,  # good
        ord('x'): False,  # bad
    }

    if path_save.exists():
        with open(path_save, 'rb') as file:
            labels = pickle.load(file)
    else:
        labels = dict()

    n = len(list(path_dir.iterdir()))

    for i, path_image in enumerate(path_dir.iterdir()):
        print('{} of {}.'.format(i, n))

        if path_image.name in labels:
            continue

        image = cv2.imread(str(path_image))
        cv2.imshow('Labeling', image)
        key = cv2.waitKey(0)

        label = map_key_label.get(key)

        if label is not None:
            labels[path_image.name] = label
        elif key == 27:
            break

    with open(path_save, 'wb') as file:
        pickle.dump(labels, file)


def check_labels():
    path = Path(r'data\xray_dataset\annotations_1\labels_good_bad.pickle')

    with open(path, 'rb') as file:
        labels = pickle.load(file)

    print(len(labels))
    print(sum(labels.values()))
    print(list(labels.items())[0])


if __name__ == '__main__':
    labeling()
    # check_labels()
    pass
