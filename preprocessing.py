
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

cv.imwrite("he.jpg", img)


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

cv.imwrite("med_blur.jpg", med_blur_img)

