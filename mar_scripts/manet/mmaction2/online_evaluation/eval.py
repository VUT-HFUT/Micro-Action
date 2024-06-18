'''
Packaging to generate zip files for uploading, 
the default coarse-grained labels are derived from the fine-grained labels, 
if you use separately generated coarse-grained labels, be sure to modify the relevant code.
'''
import os
import pickle
import zipfile
import csv

pickle_file_path = 'online_evaluation/test_result.pickle'
csv_file_path = 'online_evaluation/prediction.csv'
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
f = open(csv_file_path, 'w', newline='')
writer = csv.writer(f)
writer.writerow(["id", "pred_label_1", "pred_label_2"])
for index, data in enumerate(datas):
    file_name = "test" + str(index).zfill(4) + '.mp4'
    fine_pred = data['pred_label'].cpu().numpy()[0]
    coarse_pred = fine2coarse(fine_pred)
    writer.writerow([file_name, str(coarse_pred), str(fine_pred)])
f.close()

with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.write(csv_file_path, os.path.basename(csv_file_path))