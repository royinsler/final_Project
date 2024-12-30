import pandas as pd
import os

# train_untagged_list = []
# for f in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/'):
#     name, ext = os.path.splitext(f)
#     if ext == '.png':
#         train_untagged_list.append(f)
df_pos_scans = pd.read_excel('/mnt/md0/royi/final_Project/Preprocessing/test.xlsx' ,dtype={'test':str})
i=0

for slice in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Positive/'):
     for s in df_pos_scans["test"] :
        if slice == s and s not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive/'):
          i += 1
          os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Positive/' + slice + ' /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive/')
    # else:
    #     os.system('mv /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/' + slice + ' /mnt/data/soroka_tomo/segmented_DBT_slices_soroka//Negative/')
print(i)