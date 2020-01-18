import numpy as np
from pathlib import Path
import pickle
import shutil


def select(path_metrics, path_images, path_dir_save_selected, path_dir_save_unselected=None):
    path_model = Path(__file__).parent.parent / Path('data/models/regr.pickle')

    with open(path_metrics, 'rb') as file:
        metrics = pickle.load(file)
    with open(path_model, 'rb') as file:
        model = pickle.load(file)

    for name, metric in metrics.items():

        x = np.array([metric])
        pred = model.predict(x)

        path_image = path_images / name
        if pred == 1:
            path_save = path_dir_save_selected / name
            shutil.copyfile(path_image, path_save)
        elif path_dir_save_unselected:
            path_save = path_dir_save_unselected / name
            shutil.copyfile(path_image, path_save)
