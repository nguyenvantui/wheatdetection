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
import yaml

SEED = 99
PATH_CONFIG = "configs/config_full.yaml"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

print("Path config:", PATH_CONFIG)

with open(PATH_CONFIG) as f:
    configs = yaml.load(f)

def val():
    val_loss = 100
    val_map = 100
    return val_loss, val_map

def train():
    pass

def main():
    for epoch in range(configs["EPOCHS"]):
        print("Starting epoch:", e)
if __name__ == "__main__":
    main()