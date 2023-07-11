# %% 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests
import cv2
import json
import io

# BOX_COLOR = (255, 0, 0) # Red
BOX_COLOR = {
    'pad': (222, 3, 75), 
    'scribe_line': (178, 204, 157), 
    'contamination': (220, 3, 164), 
    'corrosion': (110, 176, 47), 
    'crack': (132, 100, 151), 
    'chip': (20, 230, 12), 
    'discolor': (90, 132, 236), 
    'particle': (232, 174, 55), 
    'scratch': (2, 152, 197), 
    'burr': (145, 206, 209), 
    'fly_die': (31, 161, 86), 
    'shift_sl': (212, 220, 104), 
    'flake': (101, 250, 98)
}

TEXT_COLOR = (255, 255, 255) # White

API = "http://localhost:6092/detection"


def visualize_bbox(img, bbox, class_name, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    # x_min, x_max, y_min, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    color = BOX_COLOR[class_name]
    cv2.rectangle(
        img, 
        (x_min, y_min), 
        (x_max, y_max), 
        color=color, 
        thickness=thickness
    )

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)    
    
    cv2.rectangle(
        img, 
        (x_min, y_min - int(1.3 * text_height)), 
        (x_min + text_width, y_min), 
        color, 
        -1
    )
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.9, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def get_bboxes(img): 
    
    # with open(img, "rb") as fs: 
    #     image = fs.read()

    files = {"file": img.getvalue()}
    response = requests.post(API, files=files)
    detection = response.json()
    # print(detection)

    bboxes = []

    detection_dict = {
        "class": [], 
        "bounding_box": []
    }

    for i in range(len(detection["classes"])): 

        bbox = detection["boxes"][i]
        det = detection["classes"][i]

        bbox.append(det)
        # print(bbox)
        bboxes.append(bbox)

    # print(bboxes)

    return bboxes


def visualize(img):

    array = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # img = image.copy()
    # img = cv2.imread(img)

    bboxes = get_bboxes(img)
    # print(bboxes)

    detection_df = pd.DataFrame(bboxes, columns=["x", "y", "w", "h", "class"])
    detection_df = detection_df[["class", "x", "y", "w", "h"]]

    for bbox in bboxes:
        image = visualize_bbox(image, bbox[:-1], bbox[-1])
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)

    return image, detection_df

if __name__ == "__main__": 

    filepath = r"C:\Users\40501\Documents\USM\Sem 4\CDS590\Image\SDSM_AOI\annotation\B5nm-6228X11822-61um-BG_DP1672999_10_2_11_BG08_Contamination_13.jpg"

    with open(filepath, "rb") as fh:
        buf = io.BytesIO(fh.read())

    image, df = visualize(buf)
    # print(type(image))

    df



# %%
