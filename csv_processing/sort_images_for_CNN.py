import os
import pandas as pd
import shutil

# make cnn folder
if not os.path.exists(os.path.join("DRDC_data", "CNN")):
    os.mkdir(os.path.join("DRDC_data", "CNN"))


# the names test, train, validation are same as the csvs (the csvs were train_1, it was renamed to train)
# the images folder are referred as f"{category}_images", for validation, it was val_images, it was renamed to validation_images
for category in ["test", "train", "validation"]:
    csv = pd.read_csv(f"csv_processing/{category}.csv")

    if not os.path.exists(os.path.join("DRDC_data", "CNN", f"{category}", "0")):
        for i in range(0, 5):
            os.mkdir(os.path.join("DRDC_data", "CNN", f"{category}", f"{i}"))
        
    for i in range(0,5):
        for image in csv.loc[csv['diagnosis'] == i, "id_code"] + '.png':
            shutil.copy2(os.path.join("DRDC_data", "preprocessed" , f"{category}_images", f"{image}"), os.path.join("DRDC_data", "CNN", f"{category}", f"{i}"))
        
    
