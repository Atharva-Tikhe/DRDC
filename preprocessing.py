
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 as cv
from matplotlib import pyplot as plt

# Preprocessing
# 1. Region of Interest (ROI) 
    # 1. Histogram equalization
    # 2. Median blur


def he_and_mb(path, write: bool, *write_path):
    '''Perform step one of preprocessing
    RGB image is read and histogram equalization is done after RGB is converted to HSV (need brightness channel)
    The HE image is then blurred using median blur (kernel - 5)
    '''
    
    # Image has to be grayscale
    # img = cv.imread('/kaggle/input/resized-2015-2019-diabetic-retinopathy-detection/resized_test19/003f0afdcd15.jpg', cv.IMREAD_GRAYSCALE)
    # equ = cv.equalizeHist(img)
    # cv.imwrite("/kaggle/working/test_he.jpg", equ)


    # convert RGB to HSV to get brightness channel (V-channel)
    img_hsv = cv.cvtColor(cv.imread(path), cv.COLOR_RGB2HSV)

    # Histogram equalisation on the V-channel
    img_hsv[:, :, 2] = cv.equalizeHist(img_hsv[:, :, 2])

    # convert image back from HSV to RGB
    img = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)

    med_blur_img = cv.medianBlur(img, 5)
    
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
    if write == True:
        # cv.imwrite(f"{write_path[0]}/he.jpg", img)
        cv.imwrite(f"{write_path[0]}/med_blur.jpg", med_blur_img)
    return med_blur_img
    

hemb_img = he_and_mb('test_images/0ad36156ad5d.jpg', True, "./test_images")



"""IMAGE SEGMENTATION 
1. Image is converted to RGB
2. Reshape to 2D
3. K-means clustering
"""

# flatten the image while preserving 3 channels
pixel_vals = hemb_img.reshape((-1,3))

pixel_vals = np.float32(pixel_vals)

#the below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
#becomes 85%
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# then perform k-means clustering with number of clusters defined as 3
#also random centres are initially choosed for k-means clustering
k = 3
retval, labels, centres = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centres = np.uint8(centres)
segmented_data = centres[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((hemb_img.shape))

cv.imwrite("./test_images/clustered_img.jpg", segmented_image)

lowest_centre = min(centres.flatten())

input_for_thresh = cv.cvtColor(segmented_image, cv.COLOR_RGB2GRAY)

thresh = cv.threshold(input_for_thresh, lowest_centre, 255, cv.THRESH_BINARY)

cv.imwrite('./test_images/thresh.jpg', thresh[1])

