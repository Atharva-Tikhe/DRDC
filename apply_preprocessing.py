import os
from preprocessing import preprocessing
import cv2 as cv


resized_19_path = os.path.join("..", "DRDC_data", "archive", "resized_test19")
resized_traintest15_path = os.path.join("..", "DRDC_data", "archive", "resized_traintest15_train19")

resized_19 = os.listdir(resized_19_path)
resized_traintest15 = os.listdir(resized_traintest15_path)

pp_resized_19_path = os.path.join("..", "DRDC_data", "preprocessed", "resized_test19")
pp_resized_traintest15 = os.path.join("..", "DRDC_data", "preprocessed", "resized_traintest15_train19")

if not os.path.exists(os.path.join("..", "DRDC_data", "preprocessed")):
    os.mkdir(os.path.join("..", "DRDC_data", "preprocessed"))
if not os.path.exists(pp_resized_19_path):
    os.mkdir(pp_resized_19_path)
if not os.path.exists(pp_resized_traintest15):
    os.mkdir(pp_resized_traintest15)


for image in resized_19:
    print(f'processing {image}')
    processed_img = preprocessing.sub_local_mean_color(os.path.join(resized_19_path, image))
    cv.imwrite(os.path.join("..", "DRDC_data", "preprocessed", "resized_test19", image), processed_img)


for image in resized_traintest15:
    print(f"processing {image}")
    processed_img = preprocessing.sub_local_mean_color(os.path.join(resized_traintest15_path, image))
    cv.imwrite(os.path.join("..", "DRDC_data", "preprocessed", "resized_traintest15_train19", image), processed_img)


