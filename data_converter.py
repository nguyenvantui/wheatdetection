import os
import cv2
import pandas as pd

import sys
import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
# from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
import os
from ipdb import set_trace as bp

SEED = 99

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

data_root = "data/"
file_path = os.path.join(data_root, 'train.csv')
marking = pd.read_csv(file_path)

print(">>> Vite Vite <<<")

print(marking.shape)

bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

# print(bboxs["width"])

for i, column in enumerate(['x', 'y', 'w', 'h']):
    marking[column] = bboxs[:,i]

marking.drop(columns=['bbox'], inplace=True)

# print("Hello marking:")
# print(marking)

# bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

# # print(bboxs["width"])

# for i, column in enumerate(['x', 'y', 'w', 'h']):
#     print(column)
#     marking[column] = bboxs[:,i]

# marking.drop(columns=['bbox'], inplace=True)
# print(marking)

for i in range(marking.shape[0]):
    print(marking.iloc[i]['image_id'])