# %%
import os
import shutil
import json
import pandas as pd
import albumentations as A
import random
from loguru import logger
import cv2
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import statistics

GOAL = 80
SPLIT = {
    "val" : 20, 
    "test": 10,
}
IMG_PATH = r"cropped_image\train"

RANDOM_STATE = 0
random.seed(RANDOM_STATE)


# %% Image augmentation to balance out the image classes
for category in os.listdir(IMG_PATH): 
    if os.path.isdir(os.path.join(IMG_PATH, category)): 

        files = [item for item in os.listdir(os.path.join(IMG_PATH, category)) if os.path.isfile(os.path.join(IMG_PATH, category, item))]

        cnt = GOAL - len(files)

        while cnt > 0: 

            logger.info(f"{category}: {cnt} images remaining. ")
            file = random.choice(files)

            file_path = os.path.join(IMG_PATH, category, file)

            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            transform = A.Compose([
            A.HorizontalFlip(p=0.5), 
            A.RandomBrightnessContrast(p=0.5), 
            A.RandomGamma(p=0.5)
            ])

            transformed = transform(image=img)

            transformed_img_file = f"AUG{cnt}_{file}"
            transformed_img_path = os.path.join(os.path.join(IMG_PATH, category), transformed_img_file)

            if cv2.imwrite(transformed_img_path, cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2RGB)): 
                cnt -= 1
            else: 
                logger.warning("Fail to save image")
                raise



        
# %% train test split

for category in os.listdir(IMG_PATH): 
    if os.path.isdir(os.path.join(IMG_PATH, category)): 
        files = [item for item in os.listdir(os.path.join(IMG_PATH, category)) if os.path.isfile(os.path.join(IMG_PATH, category, item))]
        for split, value in SPLIT.items(): 

            logger.info(split)

            img_folder = os.path.join("cropped_image", split, category)
            if not os.path.exists(img_folder): 
                os.mkdir(img_folder)
            
            while value > 0: 
                file = random.choice(files)

                shutil.move(os.path.join(IMG_PATH, category, file), os.path.join(img_folder, file))
                files.remove(file)

                value -= 1
            

# %% image resizing

# get median image size. 
if not os.path.exists(os.path.join("resized_image")): 
    os.mkdir("resized_image")

img_heights = []
img_widths = []

for root, dirs, files in os.walk("cropped_image"): 
    if len(files) != 0: 
        for file in files: 

            if file.endswith((".jpg", ".jpeg", ".bmp")): 

                image_full_path = os.path.join(root, file)

                img = cv2.imdecode(np.fromfile(image_full_path, dtype=np.uint8), -1)
                img_height = img.shape[0]
                img_width = img.shape[1]

                img_heights.append(img_height)
                img_widths.append(img_width)

height = int(statistics.median(img_heights))
width = int(statistics.median(img_widths))

# resize
transform = A.Compose([
            A.Resize(height, width, always_apply=True)
            ])

for root, dirs, files in os.walk("cropped_image"): 
    if len(files) != 0: 
        for file in files: 

            if file.endswith((".jpg", ".jpeg", ".bmp")): 

                image_full_path = os.path.join(root, file)

                img = cv2.imread(image_full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                transformed = transform(image=img)
                transformed_img_path = os.path.join("resized_image", *image_full_path.split("\\")[1:])

                transformed_img_folder = os.path.split(transformed_img_path)[0]
                while not os.path.exists(transformed_img_folder): 
                    try: 
                        os.mkdir(transformed_img_folder)
                    except: 
                        transformed_img_folder = os.path.split(transformed_img_folder)[0]

                cv2.imwrite(transformed_img_path, cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2RGB))



# %%
