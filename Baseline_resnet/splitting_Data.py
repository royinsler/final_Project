import os
import pandas as pd
import os
#from openpyxl.workbook import Workbook
pos_list = os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Positive/')
# pos_list = os.listdir(r'..\Pos\\')
#
# #pos_subjects = pos_subjects['study_ID']
# uniqe_pos = []
# for name in pos_list:
#     if name[:6] not in uniqe_pos:
#         uniqe_pos.append(name[:6])
#
neg_list = os.listdir("/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Negative/")
# # neg_list = os.listdir(r"..\Neg\\")
#
# uniqe_neg = []
# for name in neg_list:
#     if name[:6] not in uniqe_neg:
#         uniqe_neg.append(name[:6])
#
#
# #making sure slices from the same subject won't be divided into more than one dataset
# import random
# random.shuffle(uniqe_pos)
# train_pos = uniqe_pos[:round(0.7*len(uniqe_pos))]
# val_pos = uniqe_pos[round(0.7*len(uniqe_pos)):round(0.85*len(uniqe_pos))]
# test_pos = uniqe_pos[round(0.85*len(uniqe_pos)):]
#
# train_neg = []
# val_neg = []
# test_neg = []
# num_train_neg = round(0.7*len(uniqe_neg))
# num_val_neg = round(0.15*len(uniqe_neg))
# random.shuffle(uniqe_neg)
# to_check = []
# for name in uniqe_neg:
#     if name in train_pos:
#         train_neg.append(name)
#     elif name in val_pos:
#         val_neg.append(name)
#     elif name in test_pos:
#         test_neg.append(name)
#     else:
#         to_check.append(name)
#
# for name in to_check:
#     if len(train_neg) < num_train_neg:
#         train_neg.append(name)
#     elif len(val_neg) < num_val_neg:
#         val_neg.append(name)
#     else:
#         test_neg.append(name)

# df = pd.concat([pd.Series(train_pos),pd.Series(train_neg), pd.Series(val_pos), pd.Series(val_neg), pd.Series(test_pos), pd.Series(test_neg)], ignore_index=True, axis=1)
# df.columns =['train_positive', 'train_negative', 'val_positive', 'val_negative', 'test_positive', 'test_negative']
# df.to_excel('/mnt/md0/royi/final_Project/Baseline_resnet/tomo_subjects_sets_all.xlsx',index = False)
df_ds_alloc = pd.read_excel('/mnt/md0/royi/final_Project/Baseline_resnet/tomo_subjects_sets_all.xlsx' ,dtype={'study_ID':str,'View':str,'Laterality':str,'Selected_Frames':str})

#create the train,test and val directories (with pos+neg subdirectories) and move the relevant slices into it
first_run = False
count_train = 0
count_test = 0
count_val = 0
to_check = []
#The code takes into account running it from /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/
for col in df_ds_alloc:
    print(col)
    if col == 'train_positive' or col == 'train_negative':
        continue
    for id in df_ds_alloc[col]:
        print(id)
        if type(id) == float:
            id = str(id)
        if 'train' in col:
            # for slice in pos_list:
            #     if count_train == 0:
            #         # print('mkdir')
            #         # os.system(r'mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train')
            #         os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive')
            #         count_train += 1
            #     if id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive'):
            #         os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Positive/'+slice+' Train/Positive/')
            #         # os.system(r'copy ..\Pos\\'+slice+' Train\Positive\\')
            #         # print("Pos "+slice)
            for slice in neg_list:
                if count_train == 1:
                    # print('mkdir')
                    os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Negative')
                    # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Baseline_resnet\Train\\Negative')
                    count_train += 1
                if id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Negative'):
                    os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Negative/' + slice + ' Train/Negative/')
                    # os.system(r'copy ..\\Neg\\'+slice+' Train\\Negative\\')
                    # print("Neg "+slice)
        elif 'test' in col:
            for slice in pos_list:
                if count_test == 0:
                    # print('mkdir')
                    # os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test')
                    os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test/Positive')
                    # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Baseline_resnet\Test')
                    # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Baseline_resnet\Test\Positive')
                    count_test += 1
                if id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test/Positive/'):
                    os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Positive/'+slice+' Test/Positive/')
                    # os.system(r'copy ..\Pos\\'+slice+' Test\Positive\\')
                    # print("Pos "+slice)
            for slice in neg_list:
                if count_test == 1:
                    # print('mkdir')
                    os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test/Negative')
                    # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Baseline_resnet\Test\\Negative')
                    count_test += 1
                if id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test/Negative/'):
                    os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Negative/' + slice + ' Test/Negative/')
                    # os.system(r'copy ..\\Neg\\' + slice + ' Test\\Negative\\')
                    # print("Neg "+slice)
        elif 'val' in col:
            for slice in pos_list:
                if count_val == 0:
                    # print('mkdir')
                    # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Baseline_resnet\Validation')
                    # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Baseline_resnet\Validation\Positive')
                    # os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation')
                    os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive')
                    count_val += 1
                if id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive/'):
                    os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Positive/'+slice+' Validation/Positive/')
                    # os.system(r'copy ..\Pos\\'+slice+' Validation\Positive\\')
                    # print("Pos "+slice)
            for slice in neg_list:
                if count_val == 1:
                    # print('mkdir')
                    os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Negative')
                    # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Baseline_resnet\Validation\\Negative')
                    count_val += 1
                if id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Negative/'):
                    os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Negative/' + slice + ' Validation/Negative/')
                    # os.system(r'copy ..\\Neg\\' + slice + ' Validation\\Negative\\')
        else:
            to_check.append(id)
print(to_check)
                    # print("Neg "+slice)
#need to remove unecessary slices from train positive cases
# df_pos_scans = pd.read_csv('/mnt/md0/royi/final_Project/Baseline_resnet/annotations_tomo.csv' ,dtype={'study_ID':str,'View':str,'Laterality':str,'Selected_Frames':str})
# print(df_pos_scans['View'][0])
# for frames in df_pos_scans['Selected_Frames']:
#     if frames=='none' or frames=='all':
#         continue
#     elif frames.count('-')<1:
#         mylist = [int(x) for x in '3 ,2 ,6 '.split(',') if x.strip().isdigit()]
#         print(mylist)





