import pandas as pd
import os

train_pos_list = os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive')
val_pos_list = os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive')
# merged_pos_list = train_pos_list + val_pos_list
# train_pos_list = os.listdir(r'Train\Positive')
# train_pos_list = df[train_pos_list]
df_pos_scans = pd.read_csv('/mnt/md0/royi/final_Project/Baseline_resnet/annotations_tomo.csv' ,dtype={'study_ID':str,'View':str,'Laterality':str,'Selected_Frames':str})
# df_pos_scans = pd.read_csv(r'annotations_tomo.csv' ,dtype={'study_ID':str,'View':str,'Laterality':str,'Selected_Frames':str})

print(df_pos_scans['View'][0])
removed = []
to_check=[]
dont_remove = []
num_slices=0
for i, name in enumerate(df_pos_scans['study_ID']):
    print(name)
    if df_pos_scans['Selected_Frames'][i]=='none':
        print('none')
        scan_name = name+'_'+df_pos_scans['Laterality'][i]+'_'+df_pos_scans['View'][i]
        for slice in train_pos_list:
            # num_slices += 1
            if scan_name in slice:
                # print(slice)
                removed.append(slice)
                os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive/'+slice)
        for slice in val_pos_list:
            # num_slices += 1
            if scan_name in slice:
                # print(slice)
                removed.append(slice)
                os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive/'+slice)
    elif df_pos_scans['Selected_Frames'][i]=='all':
        continue
    elif df_pos_scans['Selected_Frames'][i].count('-')<1:
        print(',')
        for slice in train_pos_list:
            # num_slices += 1
            if name in slice:
                # print(df_pos_scans['Selected_Frames'][i])
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
                    os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive/'+slice)
        for slice in val_pos_list:
            # num_slices += 1
            if name in slice:
                # print(df_pos_scans['Selected_Frames'][i])
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
                    os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive/'+slice)
    elif df_pos_scans['Selected_Frames'][i].count('-') == 1:
        print('-')
        for slice in train_pos_list:
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
                    # os.system(r'rd Train\Positive\\'+slice)
                    os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/Positive/'+slice)
        for slice in val_pos_list:
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
                    # os.system(r'rd Train\Positive\\'+slice)
                    os.system('rm /mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation/Positive/'+slice)
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



