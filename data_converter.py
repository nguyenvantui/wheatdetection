import pandas as pd
import numpy as np
import cv2
import os
import re
from PIL import Image
from matplotlib import pyplot as plt
import xml
from tqdm import tqdm
import glob

obj_xml = '''
    <object>
        <name>{}</name>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>\
'''

ann_xml = '''\
<annotation>
    <filename>{}</filename>
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>3</depth>
    </size>{}
</annotation>\
'''

ROOT_OUTPUT_PATH_GENERATOR = "data/COCO_WHEAT_DATASET"
IMGDIR = ROOT_OUTPUT_PATH_GENERATOR + '/JPEGImages/'
LBLDIR = ROOT_OUTPUT_PATH_GENERATOR + '/Annotations/'
IMGSET = ROOT_OUTPUT_PATH_GENERATOR + '/ImageSets/'
all_folder = [IMGDIR, LBLDIR, IMGSET]

for folder in all_folder:
    if os.path.isdir(folder) == False:
        os.mkdir(folder)

DIR_INPUT = 'data'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
print("Path to train:", DIR_TRAIN)
print("Path to test:", DIR_TEST)

data_frame = pd.read_csv(f'{DIR_INPUT}/train.csv')
data_frame['x'] = -1
data_frame['y'] = -1
data_frame['w'] = -1
data_frame['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

data_frame[['x', 'y', 'w', 'h']] = np.stack(data_frame['bbox'].apply(lambda x: expand_bbox(x)))
data_frame.drop(columns=['bbox'], inplace=True)
data_frame['x'] = data_frame['x'].astype(np.float)
data_frame['y'] = data_frame['y'].astype(np.float)
data_frame['w'] = data_frame['w'].astype(np.float)
data_frame['h'] = data_frame['h'].astype(np.float)

image_ids = data_frame['image_id'].unique()
print("Len:", int(len(image_ids)*0.2))
print("Image ids:", image_ids)
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]

valid_df = data_frame[data_frame['image_id'].isin(valid_ids)]
data_frame = data_frame[data_frame['image_id'].isin(train_ids)]

def convert_csv_to_coco(all_id_image):

    set_path = IMGSET + "train.txt"
    fo = open(set_path, "w+", encoding="utf-8")

    for image_id in tqdm(all_id_image):
        records = data_frame[data_frame['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values

        xml = ''
        for box in boxes:
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            xml += obj_xml.format("1", box[0], box[1], box[2], box[3])
        # LBLDIR 
        # xml += obj_xml.format("1", boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
        img_data = cv2.imread("data/train/"+ image_id + ".jpg")
        height, width ,c = img_data.shape
        xml_name = image_id + ".xml"
        img_name = image_id + ".jpg"

        # saving image
        cv2.imwrite(IMGDIR + img_name, img_data)
        xml = ann_xml.format(img_name, *(width, height), xml)
        with open(LBLDIR + xml_name, 'w+', encoding="utf-8") as f:
            f.write(xml)    
        fo.writelines(image_id + "\n")
    # pass

if __name__ == "__main__":
    convert_csv_to_coco(image_ids)