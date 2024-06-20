# %%

import os
import shutil
import json
import pandas as pd
# from sklearn.model_selection import train_test_split
import albumentations as A
import random
from loguru import logger
import cv2
import numpy as np
import matplotlib.pyplot as plt

GOAL = 50
REGION_ATTRIBUTE = "class"
IMG_PATH = r"images\augmentation"
VIA_PATH = r"images\annotation.json"

RANDOM_STATE = 0
random.seed(RANDOM_STATE)


# %%
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

sample_output = {
    "_via_settings":{"ui":{"annotation_editor_height":25,"annotation_editor_fontsize":0.8,"leftsidebar_width":18,"image_grid":{"img_height":80,"rshape_fill":"none","rshape_fill_opacity":0.3,"rshape_stroke":"yellow","rshape_stroke_width":2,"show_region_shape":True,"show_image_policy":"all"},"image":{"region_label":"class","region_color":"class","region_label_font":"10pxSans","on_image_annotation_editor_placement":"NEAR_REGION"}},"core":{"buffer_size":18,"filepath":{},"default_filepath":""},"project":{"name":"via_project_val.5.12"}},
    "_via_img_metadata": {}, 
    "_via_attributes":{"region":{REGION_ATTRIBUTE:{"type":"radio","description":"","options":{},"default_options":{}}},"file":{}}
}


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes):
    img = image.copy()
    for bbox in bboxes:
        class_name = bbox[-1]
        img = visualize_bbox(img, bbox[:-1], class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

def regions(bboxes): 
    regions = []
    for bbox in bboxes: 
        region = {
            "shape_attributes": {
                "name": "rect", 
                "x": int(bbox[0]), 
                "y": int(bbox[1]),
                "width": int(bbox[2]), 
                "height": int(bbox[3])
            }, 
            "region_attributes": {
                REGION_ATTRIBUTE: bbox[-1]
            }
        }
        regions.append(region)
    
    return regions

def generate_annotation(filename, bboxes, transformed_img_path):
    filesize = os.stat(transformed_img_path).st_size 

    annot = {
        f"{filename}{filesize}": {
            "filename": filename, 
            "size": filesize, 
            "regions": regions(bboxes),
            "file_attributes": {}
        }
    }

    return annot

def get_features(via_json: dict): 
    feature = []
    print(via_json.keys())
    via = via_json["_via_img_metadata"] if "_via_img_metadata" in via_json.keys() else via_json
    # print(via_json.keys())
    # print(via)
    for key, value in via.items(): 
        print("key")
        print(key)
        for region in value["regions"]: 
            feat = region["region_attributes"][REGION_ATTRIBUTE]
            if feat not in feature:
                feature.append(feat)

    return feature



# %%
with open(VIA_PATH, "r") as fs: 
    annot = json.load(fs)["_via_img_metadata"]

features = get_features(annot)

# update via output
sample_output["_via_attributes"]["region"]["options"] = {feature: feature for feature in features}

image_files = {
    "filekey":  [],
    "filename": [], 
    "filesize": [], 
    "annot": []
}

for feature in features: 
    image_files.update({feature:[]})


for key, value in annot.items(): 
    print(value)

    features_count = {feature: 0 for feature in features}

    bbox = []

    for region in value["regions"]: 
        feat = region["region_attributes"][REGION_ATTRIBUTE]
        features_count[feat] += 1

        coor = region["shape_attributes"]

        bbox.append([coor["x"], coor["y"], coor["width"], coor["height"], feat])

    # print(features_count)

    image_files["filekey"].append(key)
    image_files["filename"].append(value["filename"])
    image_files["filesize"].append(value["size"])
    image_files["annot"].append(json.dumps(bbox))
    for feature, cnt in features_count.items(): 
        image_files[feature].append(cnt)

image_files
# %%
image_files_df = pd.DataFrame(image_files)
# image_files_df = pd.read_csv("annotationsv2.csv")

# %%

final_annotation = []

goal =  GOAL

# print(feature)
# print(goal)
# print(image_files[feature])

# create albumentations pipeline

# get relavant file names
feature_images_df = image_files_df
# feature_images_df = image_files_df[image_files_df[feature] != 0]
# feature_images_df.reset_index(drop=True, inplace=True)

# feature_images_df = image_files_df.copy(deep=True)
# feature_count = feature_images_df[feature].sum()

# print(len(feature_images_df))

aug = goal

logger.info(f"Feature: {feature}, Goal: {goal}, Images to augment: {aug}")

img_save_path = os.path.join(IMG_PATH, "augmentation")

# print(aug)
if not os.path.exists(img_save_path): 
    os.makedirs(img_save_path)
    logger.info(f"Path for {feature} created. ")

_via_img_metadata = {}

while aug > 0: 

    try: 

        # randomly choose files to perform augmentation
        key = random.randint(0, len(feature_images_df)-1)
        filename = feature_images_df.loc[key, "filename"]
        annots = json.loads(feature_images_df.loc[key, "annot"])

        print(filename)
        # print(annots)

        file_path = os.path.join(IMG_PATH, filename)

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_height = img.shape[0]
        img_width = img.shape[1]

        defects = [annot[-1] for annot in annots]
        print(defects)

        # bbox = [annot for annot in annots if feature in annot]
        bbox = annots
        
        for box in bbox: 
            for i in range(3): 
                if box[i] < 0: 
                    box[i] = 0

        print(bbox)

        # augmentation pipeline
        transform = A.Compose([
            A.BBoxSafeRandomCrop(erosion_rate=0), 
            A.VerticalFlip(p=0.5), 
            A.HorizontalFlip(p=0.5), 
            A.RandomBrightnessContrast(p=0.5), 
            A.RandomGamma(p=0.5)
        ], bbox_params=A.BboxParams(format="coco"))

        transformed = transform(image=img, bboxes=bbox)

        # visualize(
        #     transformed["image"],
        #     transformed["bboxes"]
        # )
        transformed_img_file = f"{aug}_{feature}_{filename}"
        transformed_img_path = os.path.join(img_save_path, transformed_img_file)

        if cv2.imwrite(transformed_img_path, cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2RGB)): 
            
            transformed_annotation = generate_annotation(transformed_img_file, transformed["bboxes"], transformed_img_path)
            _via_img_metadata.update(transformed_annotation)
            aug -= 1

    except Exception as e: 
        print(repr(e))

        break

final_json = sample_output

final_json["_via_img_metadata"] = _via_img_metadata

final_json_path = os.path.join(img_save_path, "augmentation.json")

with open(final_json_path, "w") as f:
    json.dump(final_json, f)


# %% join all json and move all files into one new folder
import shutil

final_annotation = annot.copy()

if not os.path.exists("final"):
    os.mkdir("final")

final_folder = os.path.join("final", IMG_PATH)

if not os.path.exists(final_folder): 
    for i in range(len(IMG_PATH.split(os.sep))): 
        folder = os.path.join("final", *IMG_PATH.split(os.sep)[:i+1])
        try: 
            os.mkdir(folder)
        except Exception as e: 
            raise e


for top, dirs, files in os.walk(IMG_PATH): 
    for file in files: 
        if ".jpg" or ".bmp" in file.lower(): 
            from_file = os.path.join(top, file)
            to_file = os.path.join(final_folder, file)
            shutil.copy(from_file, to_file)


annot_file = os.path.join(IMG_PATH, "augmentation", "augmentation.json")

with open(annot_file, "r") as f:
    aug_annot = json.load(f)["_via_img_metadata"]

final_annotation.update(aug_annot)

final_json = sample_output

final_json["_via_img_metadata"] = final_annotation

final_json_path = os.path.join(final_folder, "final.json")

with open(final_json_path, "w") as f:
    json.dump(final_json, f)



# %%
# # downsample the majority class
# # image_files_df = pd.read_csv("annotationsv2.csv")
# logger.info("Initiating downsampling")

# for feature in feature: 

#     goal =  GOAL

#     # print(feature)
#     # print(goal)
#     # print(image_files[feature])

#     # create albumentations pipeline

#     # get relavant file names
#     feature_images_df = image_files_df[image_files_df[feature] != 0]
#     # feature_images_df.reset_index(drop=True, inplace=True)
#     feature_count = feature_images_df[feature].sum()
#     # print(len(feature_images_df))

#     downsample = feature_count - goal

#     logger.info(f"Feature: {feature}, Goal: {goal}, Object count to downsample: {downsample}")

#     img_save_path = os.path.join(r"augmentation\images", feature)

#     # print(aug)
#     if not os.path.exists(os.path.join(r"augmentation\images", feature)): 
#         os.makedirs(img_save_path)
#         logger.info(f"Path for {feature} created. ")

#     _via_img_metadata = {}

#     while downsample > 0: 
#         # randomly choose files to perform augmentation
#         # key = random.randint(0, len(feature_images_df)-1)
#         key = random.choice(feature_images_df.index.to_list())
#         filename = image_files_df.at[key, "filename"]
#         annots = json.loads(image_files_df.at[key, "annot"])

#         # print(filename)
#         # print(annots)

#         indexes = [i for i in range(len(annots)) if feature in annots[i]]

#         if len(indexes)>0: 

#             drops = random.choices(indexes, k=random.randint(0, len(indexes)-1))

#             drops = set(drops)
            
#             for drop in drops:
#                 annots.pop(drop)

#             image_files_df.at[key, "annot"] = json.dumps(annots)
#             image_files_df.at[key, feature] = image_files_df.at[key, feature] - len(drops)

#             downsample -= len(drops)
    

# image_files_df.to_csv("downsample.csv", index=False)




# %%