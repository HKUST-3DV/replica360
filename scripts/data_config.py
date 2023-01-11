import numpy as np
import seaborn as sns
import scipy.io as sio
import os


class ReplicaXRDatasetConfig(object):
    def __init__(self):
        self.num_class = 9

        # self.type2class = {'basket': 0, 'bed': 1, 'cabinet': 2, 'chair': 3, 'sofa': 4, 'table': 5,
        #                    'door': 6, 'window': 7, 'bookshelf': 8, 'picture': 9,
        #                    'counter': 10, 'blinds': 11, 'desk': 12,
        #                    'shelves': 13, 'curtain': 14, 'dresser': 15, 'pillow': 16, 'mirror': 17,
        #                    'floor_mat': 18, 'clothes': 19, 'books': 20, 'fridge': 21, 'tv': 22,
        #                    'paper': 23, 'towel': 24, 'shower_curtain': 25, 'box': 26,
        #                    'whiteboard': 27, 'person': 28, 'nightstand': 29, 'toilet': 30,
        #                    'sink': 31, 'lamp': 32, 'bathtub': 33, 'bag': 34, 'stool': 35, 'rug': 36, 'top_lamp': 37, 'trash_can': 38,
        #                    'wall_clock': 39}
        self.type2class = {'basket': 0, 'bed': 1, 'cabinet': 2, 'chair': 3, 'sofa': 4, 'table': 5,
                           'door': 6, 'window': 7,  'picture': 8,  'desk': 9,
                           'shelves': 10, 'curtain': 11,  'pillow': 12, 'mirror': 13,
                           'fridge': 14, 'tv': 15, 'towel': 16, 'box': 17,
                           'nightstand': 18, 'toilet': 19,
                           'sink': 20, 'lamp': 21, 'stool': 22, 'rug': 23, 'trash_can': 24,
                           'wall_clock': 25}


REPLICA_PANO_26_CLASSES = list(ReplicaXRDatasetConfig().type2class.keys())

REPLICA_PANO_29_CLASSES = REPLICA_PANO_26_CLASSES + \
    ['walls', 'floors', 'ceilings']


# colorbox_path = 'external/cooperative_scene_parsing/evaluation/vis/igibson_colorbox.mat'
replicapano_colorbox = np.array(sns.hls_palette(
    n_colors=len(REPLICA_PANO_29_CLASSES), l=.45, s=.8))
# if not os.path.exists(colorbox_path):
#     sio.savemat(colorbox_path, {'igibson_colorbox': igibson_colorbox})
