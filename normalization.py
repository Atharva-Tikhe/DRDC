import cv2 as cv 

img = cv.imread("test_images/graham.jpg")
gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# img = cv.resize(img, (512,512))
mean, stddev = cv.meanStdDev(gray_img)

norm_img = (img - mean)/stddev

cv.imwrite("test_images/normalized.jpg" , norm_img)
