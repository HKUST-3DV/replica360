import numpy as np
import seaborn as sns
import scipy.io as sio
from typing import List, Sequence, Tuple, Union


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


d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)


# [d3_40_colors_hex]
d3_40_colors_hex: List[str] = [
    "0x1f77b4",
    "0xaec7e8",
    "0xff7f0e",
    "0xffbb78",
    "0x2ca02c",
    "0x98df8a",
    "0xd62728",
    "0xff9896",
    "0x9467bd",
    "0xc5b0d5",
    "0x8c564b",
    "0xc49c94",
    "0xe377c2",
    "0xf7b6d2",
    "0x7f7f7f",
    "0xc7c7c7",
    "0xbcbd22",
    "0xdbdb8d",
    "0x17becf",
    "0x9edae5",
    "0x393b79",
    "0x5254a3",
    "0x6b6ecf",
    "0x9c9ede",
    "0x637939",
    "0x8ca252",
    "0xb5cf6b",
    "0xcedb9c",
    "0x8c6d31",
    "0xbd9e39",
    "0xe7ba52",
    "0xe7cb94",
    "0x843c39",
    "0xad494a",
    "0xd6616b",
    "0xe7969c",
    "0x7b4173",
    "0xa55194",
    "0xce6dbd",
    "0xde9ed6",
]
# [/d3_40_colors_hex]