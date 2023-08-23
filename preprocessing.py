
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 as cv
from matplotlib import pyplot as plt

# Preprocessing
# 1. Region of Interest (ROI) 
    # 1. Histogram equalization
    # 2. Median blur



# Image has to be grayscale
# img = cv.imread('/kaggle/input/resized-2015-2019-diabetic-retinopathy-detection/resized_test19/003f0afdcd15.jpg', cv.IMREAD_GRAYSCALE)
# equ = cv.equalizeHist(img)
# cv.imwrite("/kaggle/working/test_he.jpg", equ)

# Keep color
rgb_img = cv.imread('../DRDC_data/archive/resized_test19/0ad36156ad5d.jpg')

# convert RGB to HSV to get brightness channel (V-channel)
img_hsv = cv.cvtColor(rgb_img, cv.COLOR_BGR2HSV)

# Histogram equalisation on the V-channel
img_hsv[:, :, 2] = cv.equalizeHist(img_hsv[:, :, 2])

# convert image back from HSV to RGB
img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

# cv.imwrite("he.jpg", img)


#```
#     # convert from RGB color-space to YCrCb
# ycrcb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2YCrCb)
# 
#     # equalize the histogram of the Y channel
# ycrcb_img[:, :, 0] = cv.equalizeHist(ycrcb_img[:, :, 0])
# 
#     # convert back to RGB color-space from YCrCb
# equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)
# 
# cv.imwrite('/kaggle/working/equalized_img.jpg', equalized_img)
# ```
# source: https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image

# Added a median blur to take care of salt and pepper noise or any noise for that matter.
# Advantage: the centre pixel value is one that exists in the image.

med_blur_img = cv.medianBlur(img, 5)

# cv.imwrite("med_blur.jpg", med_blur_img)


"""IMAGE SEGMENTATION 
1. Image is converted to RGB
2. Reshape to 2D
3. K-means clustering
"""

rgb_image = cv.cvtColor(med_blur_img, cv.COLOR_BGR2RGB)
cv.imwrite("rgb_image.jpg",rgb_image)

pixel_vals = rgb_image.reshape((-1,3))

pixel_vals = np.float32(pixel_vals)

#the below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
#becomes 85%
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# then perform k-means clustering with number of clusters defined as 3
#also random centres are initially choosed for k-means clustering
k = 3
retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((rgb_image.shape))

cv.imwrite("segmented.jpg", segmented_image)


