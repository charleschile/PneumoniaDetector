"""
segmentation.py is developed to call the segmentation strategies based on U-Net CNN model and watershed algorithm respectively.
"""

# Import libraries
from PIL import Image
from unet import Unet
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import threshold_otsu, threshold_li, threshold_local
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed

# Functions

def image_segmentation(import_img, export_img = None, segmentation_method = ''):
    """Call U-Net model or watershed algorithm according to which one the user has chosen.

    :param import_img: The input image file path.
    :param export_img: The export image file path.
    :param segmentation_method: The segmentation method user choose in the combo box.
    :return: Return boolean False if the segmentation is done.
    """
    if segmentation_method == 'U-Net model':
        unet_segmentation(import_img, export_img)
        return False
    elif segmentation_method == 'Watershed algorithm':
        watershed_segmentation(import_img, export_img)
        return False


def unet_segmentation(import_path, export_path):
    """Call the unet_segmentation function in the unet_nets.py

    :param import_path: The input image file path.
    :param export_path: The export image file path.
    :return: Return boolean False if the U-Net model has been run
    """
    unet = Unet()
    mode = "predict"
    if mode == "predict":
        while True:
            img = import_path
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image)
            if export_path is not None:
                # save the export image file
                r_image.save(export_path)
                return False
            else:
                print('no export path')


def watershed_segmentation(import_path, export_path):
    """Run the watershed segmentation on the input image.

    :param import_path: The input image file path.
    :param export_path: The export image file path.
    :return: Return boolean False if the watershed algorithm has been run.
    """

    chest = plt.imread(import_path)
    seg = chest[:, :]
    seg_threshold_local = seg > threshold_local(seg, block_size=159)
    thr_local_no_small = remove_small_objects(seg_threshold_local, min_size=0)
    thr_local_no_small_no_holes = remove_small_holes(thr_local_no_small, area_threshold=0)
    best_mask = thr_local_no_small_no_holes
    distance = distance_transform_edt(best_mask)
    local_maxima_idx = peak_local_max(distance, min_distance=0, footprint=np.ones((10, 10)))
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(local_maxima_idx.T)] = True
    markers = label(mask)

    # plot the segmented image
    segmented = watershed(-distance, markers, mask=best_mask)
    plt.imshow(segmented, alpha=1)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(export_path, bbox_inches='tight',pad_inches = 0)
    return False