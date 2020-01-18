from pathlib import Path

from masks.utils import get_masks
from selection.metrics import get_metrics
from selection.select import select


def main():
    path_project = Path(__file__).parent.parent

    tag = 'test'
    path_images = Path(r'C:\Downloads\test')

    path_save_masks = path_project / Path(r'data\pipeline\masks\masks_{}.pickle'.format(tag))
    path_save_mask_images = None  # path_project / Path(r'data\masks\mask_{}'.format(tag))
    path_metrics = path_project / Path(r'data\pipeline\metrics\metrics_{}.pickle'.format(tag))
    path_dir_save_selected = path_project / Path(r'data\pipeline\selected')
    path_dir_save_unselected = path_project / Path(r'data\pipeline\unselected')

    get_masks(path_images, path_save_masks, path_save_mask_images)
    get_metrics(path_save_masks, path_metrics)
    select(path_metrics, path_images, path_dir_save_selected, path_dir_save_unselected)


if __name__ == '__main__':
    main()
