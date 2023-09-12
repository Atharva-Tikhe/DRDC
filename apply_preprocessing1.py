import os
from preprocessing import preprocessing
import cv2 as cv


test_images_path = os.path.join("..", "DRDC_data", "archive", "test_images")
train_images_path = os.path.join("..", "DRDC_data", "archive", "train_images")
val_images_path = os.path.join("..", "DRDC_data", "archive", "val_images")

test_images = os.listdir(test_images_path)
train_images = os.listdir(train_images_path) 
val_images = os.listdir(val_images_path) 

pp_test_images_path = os.path.join("..", "DRDC_data", "preprocessed", "test_images")
pp_train_images = os.path.join("..", "DRDC_data", "preprocessed", "train_images")
pp_val_images = os.path.join("..", "DRDC_data", "preprocessed", "val_images")

if not os.path.exists(os.path.join("..", "DRDC_data", "preprocessed")):
    os.mkdir(os.path.join("..", "DRDC_data", "preprocessed"))
if not os.path.exists(pp_test_images_path):
    os.mkdir(pp_test_images_path)
if not os.path.exists(pp_train_images):
    os.mkdir(pp_train_images)
if not os.path.exists(pp_val_images):
    os.mkdir(pp_val_images)


for image in test_images:
    print(f'processing {image}')
    processed_img = preprocessing.sub_local_mean_color(os.path.join(test_images_path, image))
    cv.imwrite(os.path.join("..", "DRDC_data", "preprocessed", "test_images", image), processed_img)


for image in train_images:
    print(f"processing {image}")
    processed_img = preprocessing.sub_local_mean_color(os.path.join(train_images_path, image))
    cv.imwrite(os.path.join("..", "DRDC_data", "preprocessed", "train_images", image), processed_img)

for image in val_images:
    print(f"processing {image}")
    processed_img = preprocessing.sub_local_mean_color(os.path.join(val_images_path, image))
    cv.imwrite(os.path.join("..", "DRDC_data", "preprocessed", "val_images", image), processed_img)



