import pandas as pd
import os
#from openpyxl.workbook import Workbook


pos_list = os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Positive/')

df_pos_scans = pd.read_csv('/mnt/md0/royi/final_Project/Preprocessing/annotations_tomo.csv' ,dtype={'study_ID':str,'View':str,'Laterality':str,'Selected_Frames':str})
# df_pos_scans = pd.read_csv(r'annotations_tomo.csv' ,dtype={'study_ID':str,'View':str,'Laterality':str,'Selected_Frames':str})

#Create a list of files(removed) that shouldn't be used on train
removed = []
to_check=[]
dont_remove = []
num_slices=0
for i, name in enumerate(df_pos_scans['study_ID']):
    print(name)
    if df_pos_scans['Selected_Frames'][i]=='none':
        print('none')
        scan_name = name+'_'+df_pos_scans['Laterality'][i]+'_'+df_pos_scans['View'][i]
        for slice in pos_list:
        #     # num_slices += 1
            if scan_name in slice:
                # print(slice)
                removed.append(slice)
        #         os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive/'+slice)
        #         os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive/'+slice)
        # for slice in pos_list:
        #     # num_slices += 1
        #     if scan_name in slice:
        #         # print(slice)
        #         removed.append(slice)
        #         # os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive/'+slice)
    elif df_pos_scans['Selected_Frames'][i]=='all':
        continue
    elif df_pos_scans['Selected_Frames'][i].count('-')<1:
        print(',')
        for slice in pos_list:
        #     # num_slices += 1
            if name in slice:
        #         # print(df_pos_scans['Selected_Frames'][i])
                mylist = df_pos_scans['Selected_Frames'][i].split(',')
                mylist[0]=mylist[0][1:]
                mylist[(len(mylist)-1)]=mylist[(len(mylist)-1)][:-1]
                # print(mylist)
                mylist = [int(i) for i in mylist]
                # mylist = [int(x) for x in df_pos_scans['Selected_Frames'][i].split(',') if x.strip().isdigit()]
                # print(mylist)
                is_slice_good = False
                for slice_num in mylist:
                    # print(slice_num)
                    slice_name = name + '_' + df_pos_scans['Laterality'][i] + '_' + df_pos_scans['View'][i]+'_'+str(slice_num)+'.png'
                    if slice == slice_name:
                        dont_remove.append(slice)
                        is_slice_good = True
                if not is_slice_good and slice not in dont_remove and slice not in removed:
                    # print(slice)
                    removed.append(slice)
                        # os.system(r'rd \Train\Positive\\'+slice)
                    # os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive/'+slice)
        # for slice in val_pos_list:
        #     # num_slices += 1
        #     if name in slice:
        #         # print(df_pos_scans['Selected_Frames'][i])
        #         mylist = df_pos_scans['Selected_Frames'][i].split(',')
        #         mylist[0]=mylist[0][1:]
        #         mylist[(len(mylist)-1)]=mylist[(len(mylist)-1)][:-1]
        #         # print(mylist)
        #         mylist = [int(i) for i in mylist]
        #         # mylist = [int(x) for x in df_pos_scans['Selected_Frames'][i].split(',') if x.strip().isdigit()]
        #         # print(mylist)
        #         is_slice_good = False
        #         for slice_num in mylist:
        #             # print(slice_num)
        #             slice_name = name + '_' + df_pos_scans['Laterality'][i] + '_' + df_pos_scans['View'][i]+'_'+str(slice_num)+'.png'
        #             if slice == slice_name:
        #                 dont_remove.append(slice)
        #                 is_slice_good = True
        #         if not is_slice_good and slice not in dont_remove and slice not in removed:
        #             # print(slice)
        #             removed.append(slice)
        #             # os.system(r'rd \Train\Positive\\'+slice)
        #             os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive/'+slice)
        # for slice in pos_list:
        #     # num_slices += 1
        #     if name in slice:
        #         # print(df_pos_scans['Selected_Frames'][i])
        #         mylist = df_pos_scans['Selected_Frames'][i].split(',')
        #         mylist[0]=mylist[0][1:]
        #         mylist[(len(mylist)-1)]=mylist[(len(mylist)-1)][:-1]
        #         # print(mylist)
        #         mylist = [int(i) for i in mylist]
        #         # mylist = [int(x) for x in df_pos_scans['Selected_Frames'][i].split(',') if x.strip().isdigit()]
        #         # print(mylist)
        #         is_slice_good = False
        #         for slice_num in mylist:
        #             # print(slice_num)
        #             slice_name = name + '_' + df_pos_scans['Laterality'][i] + '_' + df_pos_scans['View'][i]+'_'+str(slice_num)+'.png'
        #             if slice == slice_name:
        #                 dont_remove.append(slice)
        #                 is_slice_good = True
        #         if not is_slice_good and slice not in dont_remove and slice not in removed:
        #             # print(slice)
        #             removed.append(slice)
                    # os.system(r'rd \Train\Positive\\'+slice)
                    # os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive/'+slice)
    elif df_pos_scans['Selected_Frames'][i].count('-') == 1:
        # print('-')
        for slice in pos_list:
            # num_slices += 1
            if name in slice:
                # print(df_pos_scans['Selected_Frames'][i])
                mylist = df_pos_scans['Selected_Frames'][i].split('-')
                # print(mylist)
                mylist[0]=mylist[0][1:]
                mylist[1]=mylist[1][:-1]
                mylist = [int(i) for i in mylist]
                start_i = mylist[0]
                end_i = mylist[1]+1
                is_slice_good = False
                for slice_num in range(start_i,end_i):
                    # print(slice_num)
                    slice_name = name + '_' + df_pos_scans['Laterality'][i] + '_' + df_pos_scans['View'][i]+'_'+str(slice_num)+'.png'
                    # print(slice_name)
                    # print(slice)
                    # print(slice_name)
                    if slice == slice_name:
                        dont_remove.append(slice)
                        is_slice_good = True
                if not is_slice_good and slice not in dont_remove and slice not in removed:
                    removed.append(slice)
                    # print(slice)
        #             # os.system(r'rd Train\Positive\\'+slice)
        #             os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive/'+slice)
        # for slice in val_pos_list:
        #     # num_slices += 1
        #     if name in slice:
        #         # print(df_pos_scans['Selected_Frames'][i])
        #         mylist = df_pos_scans['Selected_Frames'][i].split('-')
        #         # print(mylist)
        #         mylist[0]=mylist[0][1:]
        #         mylist[1]=mylist[1][:-1]
        #         mylist = [int(i) for i in mylist]
        #         start_i = mylist[0]
        #         end_i = mylist[1]+1
        #         is_slice_good = False
        #         for slice_num in range(start_i,end_i):
        #             # print(slice_num)
        #             slice_name = name + '_' + df_pos_scans['Laterality'][i] + '_' + df_pos_scans['View'][i]+'_'+str(slice_num)+'.png'
        #             # print(slice_name)
        #             # print(slice)
        #             # print(slice_name)
        #             if slice == slice_name:
        #                 dont_remove.append(slice)
        #                 is_slice_good = True
        #         if not is_slice_good and slice not in dont_remove and slice not in removed:
        #             removed.append(slice)
        #             # print(slice)
        #             # os.system(r'rd Train\Positive\\'+slice)
        #             os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive/'+slice)
        # for slice in pos_list:
        #     # num_slices += 1
        #     if name in slice:
        #         # print(df_pos_scans['Selected_Frames'][i])
        #         mylist = df_pos_scans['Selected_Frames'][i].split('-')
        #         # print(mylist)
        #         mylist[0]=mylist[0][1:]
        #         mylist[1]=mylist[1][:-1]
        #         mylist = [int(i) for i in mylist]
        #         start_i = mylist[0]
        #         end_i = mylist[1]+1
        #         is_slice_good = False
        #         for slice_num in range(start_i,end_i):
        #             # print(slice_num)
        #             slice_name = name + '_' + df_pos_scans['Laterality'][i] + '_' + df_pos_scans['View'][i]+'_'+str(slice_num)+'.png'
        #             # print(slice_name)
        #             # print(slice)
        #             # print(slice_name)
        #             if slice == slice_name:
        #                 dont_remove.append(slice)
        #                 is_slice_good = True
        #         if not is_slice_good and slice not in dont_remove and slice not in removed:
        #             removed.append(slice)
                    # print(slice)
                    # os.system(r'rd Train\Positive\\'+slice)
                    # _os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive/'+slice)
    else:
        to_check.append(name)

print('to check:')
print(to_check)
# print(num_slices)
# print(len(merged_pos_list))
print('num of removed '+str(len(removed)))
df = pd.concat([pd.Series(removed)], ignore_index=True, axis=1)
df.columns =['removed_slices']
df.to_excel('removed_slices.xlsx',index = False)

#Create an excel sheet with division of studies into the different sets (Pos/Neg/Train/Validation etc...)
# pos_list = os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/New_Pos/')
# pos_list = os.listdir(r'..\Pos\\')
#
# #pos_subjects = pos_subjects['study_ID']
# uniqe_pos = []
# for name in pos_list:
#     if name[:6] not in uniqe_pos:
#         uniqe_pos.append(name[:6])
# #
neg_list = os.listdir("/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Negative/")
# # # neg_list = os.listdir(r"..\Neg\\")
# #
# uniqe_neg = []
# for name in neg_list:
#     if name[:6] not in uniqe_neg:
#         uniqe_neg.append(name[:6])
#
# make sure slices from the same subject won't be divided into more than one dataset
# import random
# random.shuffle(uniqe_pos)
# train_pos = uniqe_pos[:round(0.7*len(uniqe_pos))]
# val_pos = uniqe_pos[round(0.7*len(uniqe_pos)):round(0.85*len(uniqe_pos))]
# test_pos = uniqe_pos[round(0.85*len(uniqe_pos)):]
# #
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
#
# df = pd.concat([pd.Series(train_pos),pd.Series(train_neg), pd.Series(val_pos), pd.Series(val_neg), pd.Series(test_pos), pd.Series(test_neg)], ignore_index=True, axis=1)
# df.columns =['train_positive', 'train_negative', 'val_positive', 'val_negative', 'test_positive', 'test_negative']
# df.to_excel('/mnt/md0/royi/final_Project/Preprocessing/tomo_subjects_sets_all.xlsx',index = False)

#Load the excel sheet
df_ds_alloc = pd.read_excel('/mnt/md0/royi/final_Project/Preprocessing/tomo_subjects_sets_all.xlsx' ,dtype={'study_ID':str,'View':str,'Laterality':str,'Selected_Frames':str})

#create the train,test and val directories (with pos+neg subdirectories) and move the relevant slices into it
first_run = False
count_train = 0
count_test = 0
count_val = 0
to_check = []
#The code takes into account running it from /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/
for col in df_ds_alloc:
    print(col)
    # if col == 'train_positive' or col == 'train_negative' or col == 'test_negative'or col == 'test_positive' or col == 'val_negative':
    #     continue
    for id in df_ds_alloc[col]:
    # id = 'SD8077'
        print(id)
        if type(id) == float:
            id = str(id)
        if 'train' in col:
            classes = ['Negative', 'Positive']

            root_dir = '/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/'
            for class_idx, class_name in enumerate(classes):
                class_dir = os.path.join(root_dir, class_name)
                scan_names = [item[0:12] for item in os.listdir(class_dir)]
                unique_scan_names = list(set(scan_names))
                print("started copying to train dirs")
                print(len(unique_scan_names))
                image_count = 0
                for scan in unique_scan_names:
                    # if scan == 'SD8077_L_CC_':
                    if id in scan:
                    # print(scan)
                        i = 0
                        # for name in self.train_sub:
                        for file in os.listdir(class_dir):
                            print(file)
                            if i == 10:
                                break
                            elif scan in file:
                                print(file)
                                file_path = os.path.join(class_dir, file)
                                # image_count += 1
                                # print(image_count)
                                if class_name == 'Negative' and file not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train_small/Negative/'):
                                    print("neg addition")
                                    os.system('sudo cp ' + file_path + ' /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train_small/Negative/')
                                    i += 1
                                elif file not in removed and file not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train_small/Positive/'):
                                    print("pos addition")
                                    os.system('sudo cp ' + file_path + ' /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train_small/Positive/')
                                    i += 1
                                else:
                                    print("didnt copy")
                                    continue
                                # print(i)
                            else:
                                print("Scan not in file")
                                continue
                        print(scan + ' which is ' + class_name + ' got to ' + str(i) + ' slices')

            # i = 0
            # for slice in pos_list:
            #     if i == 10:
            #         break
            #         # print('mkdir')
            #         # os.system(r'mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train')
            #         # count_train += 1
            #     elif id in slice and slice not in removed:
            #         os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Positive/'+slice+' /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train_small/Positive/')
            #         # os.system(r'copy ..\Pos\\'+slice+' Train\Positive\\')
            #         # print("Pos "+slice)
            #         i += 1
            # print(id+' has gotten to '+str(i))
            # i = 0
            # for slice in neg_list:
            #     if i == 10:
            #         # print('mkdir')
            #         # os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train_small/Negative')
            #         # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Preprocessing\Train\\Negative')
            #         count_train += 1
            #     elif id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train_small/Negative'):
            #         os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Negative/' + slice + ' /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train_small/Negative/')
            #         # os.system(r'copy ..\\Neg\\'+slice+' Train\\Negative\\')
            #         # print("Neg "+slice)
            #         i += 1


        # elif 'test' in col:
        #     for slice in pos_list:
        #         if count_test == 0:
        #             # print('mkdir')
        #             # os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test')
        #             # os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test_small/Positive')
        #             # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Preprocessing\Test')
        #             # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Preprocessing\Test\Positive')
        #             count_test += 1
        #         if id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test_small/Positive/'):
        #             os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Positive/'+slice+' /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test_small/Positive/')
        #             # os.system(r'copy ..\Pos\\'+slice+' Test\Positive\\')
        #             # print("Pos "+slice)
        #     for slice in neg_list:
        #         if count_test == 1:
        #             # print('mkdir')
        #             # os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test/Negative')
        #             # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Preprocessing\Test\\Negative')
        #             count_test += 1
        #         if id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test_small/Negative/'):
        #             os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Negative/' + slice + ' /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test_small/Negative/')
        #             # os.system(r'copy ..\\Neg\\' + slice + ' Test\\Negative\\')
        #     #         # print("Neg "+slice)
        # elif 'val' in col:
        #     for slice in pos_list:
        #         if count_val == 0:
        #             # print('mkdir')
        #             # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Preprocessing\Validation')
        #             # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Preprocessing\Validation\Positive')
        #             # os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation')
        #             # os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation_small/Positive')
        #             count_val += 1
        #         if id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation_small/Positive/'):
        #             os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Positive/'+slice+' /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation_small/Positive/')
        #             # os.system(r'copy ..\Pos\\'+slice+' Validation\Positive\\')
        #             # print("Pos "+slice)
        #     for slice in neg_list:
        #         if count_val == 1:
        #             # print('mkdir')
        #             # os.system('mkdir /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Negative')
        #             # os.system(r'mkdir C:\Users\\royin\PycharmProjects\\final_Project\Preprocessing\Validation\\Negative')
        #             count_val += 1
        #         if id in slice and slice not in os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation_small/Negative/'):
        #             os.system('cp /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Negative/' + slice + ' /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation_small/Negative/')
        #             # os.system(r'copy ..\\Neg\\' + slice + ' Validation\\Negative\\')
        # else:
        #     to_check.append(id)
print(to_check)
                    # print("Neg "+slice)
#removal of unecessary slices from train positive cases
# df_pos_scans = pd.read_csv('/mnt/md0/royi/final_Project/Preprocessing/annotations_tomo.csv' ,dtype={'study_ID':str,'View':str,'Laterality':str,'Selected_Frames':str})
# print(df_pos_scans['View'][0])
# for frames in df_pos_scans['Selected_Frames']:
#     if frames=='none' or frames=='all':
#         continue
#     elif frames.count('-')<1:
#         mylist = [int(x) for x in '3 ,2 ,6 '.split(',') if x.strip().isdigit()]
#         print(mylist)





