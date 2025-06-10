'''
Packaging to generate zip files for uploading, 
the default coarse-grained labels are derived from the fine-grained labels, 
if you use separately generated coarse-grained labels, be sure to modify the relevant code.
'''
import os
import pickle
import zipfile
import csv
import numpy as np

pickle_file_path = 'online_evaluation/test_result.pickle'
pred_file_path = 'online_evaluation/prediction.csv'
zip_file_path = 'online_evaluation/submission.zip'

def fine2coarse(x):
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

with open(pickle_file_path, 'rb') as file:
    datas = pickle.load(file)

with open('./data/ma52/test_list_videos.txt', 'r') as f:
    file_names = [line.strip().split()[0] for line in f.readlines()]

with open(pred_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['vid'] + \
             [f'action_pred_{i}' for i in range(1, 6)] + \
             [f'body_pred_{i}' for i in range(1, 6)]
    writer.writerow(header)

    for index, data in enumerate(datas):
        file_name = file_names[index]

        pred_scores = data
        top5_fine = np.argsort(pred_scores)[-5:][::-1].tolist()

        # convert action-level label to body-level label. 
        # Note that this is a simple conversion.
        top5_coarse = [fine2coarse(x) for x in top5_fine]

        writer.writerow([file_name] + top5_fine + top5_coarse)

with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.write(pred_file_path, os.path.basename(pred_file_path))