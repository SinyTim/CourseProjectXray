import cv2
import numpy as np
from pathlib import Path


drawing = False  # true if mouse is pressed
prev_point = None


def annotation():  # speed: 7.8 images per minute.
    path_dir = Path(r'data\xray_dataset\annotations_1\18_24f')
    path_save = Path(r'data\xray_dataset\annotations_1\annotations')

    def draw(event, x, y, flags, param):
        global drawing, prev_point

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            prev_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(image, prev_point, (x, y), color=(0, 0, 255), thickness=5)
                prev_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            prev_point = None

    cv2.namedWindow('annotation')
    cv2.setMouseCallback('annotation', draw)

    n = len(list(path_dir.iterdir()))

    for i, path_image in enumerate(path_dir.iterdir()):
        print('{} of {}.'.format(i, n))
        path_save_annotation = path_save / path_image.name
        if path_save_annotation.exists():
            continue

        image = cv2.imread(str(path_image))

        while True:
            cv2.imshow('annotation', image)
            key = cv2.waitKey(1)
            if key == 32:
                break
            elif key == 27:
                exit()

        cv2.imwrite(str(path_save_annotation), image)

    cv2.destroyAllWindows()


def check_annotation():
    path_annotation = Path(r'data\xray_dataset\annotations_1\annotations\20191119-142315_648826_0.577.png')
    image = cv2.imread(str(path_annotation))
    image = image[..., 2] > 250
    image = image.astype(np.float32)
    cv2.imshow('annotation', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    annotation()
    # check_annotation()
