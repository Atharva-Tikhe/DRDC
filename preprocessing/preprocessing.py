import cv2 as cv
import numpy as np


def perform_he_mb(path, clahe: bool, write: bool, *write_path):
    '''
        Performs first task of preprocessing pipeline
        This function performs histogram equalization (increased contrast) and median blur (kernel = 5, get rid of noise)
        returns: final image
    '''
    ori_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    
    if clahe == False:
        # Perform classic histogram equalization
        he = cv.equalizeHist(ori_img)
    else:
        # Perform Contrast Limited Adaptive Histogram Equalization
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        he = clahe.apply(ori_img)

    # Perform median blur or he image (other kernel sizes aren't checked)
    med_blur_img = cv.medianBlur(he, 5)

    if write == True:
        cv.imwrite(f"{write_path[0]}/he.jpg", he)
        cv.imwrite(f"{write_path[0]}/med_blur.jpg", med_blur_img)
    return med_blur_img

def perform_thresholding(img):
    img_mat = np.asarray(img)

    threshold = np.median(np.unique(img_mat))

    thresh = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    cv.imwrite("./test_images/thresh_clahe.jpg", thresh[1])


def scaleRadius(img, scale):
    x = img[int(img.shape[0]/2), :, :].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale * 1.0/ r
    return cv.resize(img,(0,0),fx=s,fy=s)

    
def sub_local_mean_color(img_path):
    img = cv.imread(img_path)
    scale = 300
    scaled_img = scaleRadius(img, scale)
    # subtract local mean color
    scaled_img = cv.addWeighted(scaled_img, 4, cv.GaussianBlur(scaled_img, (0,0), scale/2), -4, 128)

    rem_10 = np.zeros(scaled_img.shape)
    cv.circle(rem_10, (int(scaled_img.shape[1]/2), int(scaled_img.shape[0]/2)), int(scale * 0.9), (1,1,1), -1, 8, 0)

    scaled_img = scaled_img * rem_10 + 128 * (1-rem_10)
    return scaled_img


