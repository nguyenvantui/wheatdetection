import pandas as pd
import os
import numpy as np

data_root = "data/"
file_path = os.path.join(data_root, 'train.csv')
marking = pd.read_csv(file_path)

bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'w', 'h']):
    marking[column] = bboxs[:,i]

marking.drop(columns=['bbox'], inplace=True)
print(marking)