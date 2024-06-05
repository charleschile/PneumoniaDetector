# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import threshold_otsu, threshold_li, threshold_local
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed



# Read the image
cells = plt.imread("/Users/lechi/Desktop/BIA group project/chest_try/person1944_bacteria_4869.jpeg")

# Print its shape
print(cells.shape)

# Display the image
# plt.imshow(cells)
# plt.show()
nuclei = cells[:, :]
# plt.imshow(nuclei, cmap="gray")
# plt.show()
nuclei_threshold_local = nuclei > threshold_local(nuclei, block_size=159)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15, 5))
thr_local_no_small = remove_small_objects(nuclei_threshold_local, min_size=0)
thr_local_no_small_no_holes = remove_small_holes(thr_local_no_small, area_threshold = 0)

best_mask = thr_local_no_small_no_holes
distance = distance_transform_edt(best_mask)
# plt.imshow(distance)
# plt.show()
# Find the local maxima of the distance map.
# We need to impose a minimum distance between the peaks and we can specify
# a "footprint" to search for local maxima. You can experiment with these value to find
# one that works best with your image
# peak_local_max returns the indices of the maxima
local_maxima_idx = peak_local_max(distance, min_distance=0, footprint=np.ones((10, 10)))
# We create a matrix of "False" with the same shape of the image
mask = np.zeros(distance.shape, dtype=bool)
# We now mark the maxima as True (note that we need to transpose the matrix)
mask[tuple(local_maxima_idx.T)] = True

# Label connected regions
markers = label(mask)
# Split touching nuclei using watershed
segmented = watershed(-distance, markers, mask=best_mask)
plt.imshow(segmented, alpha=1)

plt.show()


# Use label2rgb to help visualize the results
# from skimage.color import label2rgb
# segmented = remove_small_objects(segmented, min_size=1000)
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
# ax.imshow(label2rgb(segmented, bg_label=0))
# plt.show()