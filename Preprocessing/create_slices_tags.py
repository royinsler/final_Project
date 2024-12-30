import pandas as pd
import os

train_untagged_list = os.listdir('/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/')
#train_untagged_list = os.listdir(r'C:\Users\royin\PycharmProjects\final_Project\Train')

df_pos_scans = pd.read_csv('/mnt/md0/royi/final_Project/Preprocessing/annotations_tomo.csv' ,dtype={'study_ID':str,'View':str,'Laterality':str,'Selected_Frames':str})
#df_pos_scans = pd.read_csv(r'C:\Users\royin\PycharmProjects\final_Project\Preprocessing\annotations_tomo.csv' ,dtype={'study_ID':str,'View':str,'Laterality':str,'Selected_Frames':str})

print(df_pos_scans['View'][0])
pos = []
to_check=[]
for i, name in enumerate(df_pos_scans['study_ID']):
    if df_pos_scans['Selected_Frames'][i]=='none':
        continue
    elif df_pos_scans['Selected_Frames'][i] =='all':
        scan_name = name+'_'+df_pos_scans['Laterality'][i]+'_'+df_pos_scans['View'][i]
        for slice in train_untagged_list:
            if scan_name in slice:
                pos.append(slice)
    elif df_pos_scans['Selected_Frames'][i].count('-')<1:
        for slice in train_untagged_list:
            mylist = df_pos_scans['Selected_Frames'][i].split(',')
            mylist[0]=mylist[0][1:]
            mylist[(len(mylist)-1)]=mylist[(len(mylist)-1)][:-1]
            mylist = [int(i) for i in mylist]
            # mylist = [int(x) for x in df_pos_scans['Selected_Frames'][i].split(',') if x.strip().isdigit()]
            print(mylist)
            for slice_num in mylist:
                slice_name = name + '_' + df_pos_scans['Laterality'][i] + '_' + df_pos_scans['View'][i]+'_'+str(slice_num)+'.png'
                if slice == slice_name:
                    pos.append(slice)
    elif df_pos_scans['Selected_Frames'][i].count('-') == 1:
        for slice in train_untagged_list:
            print(df_pos_scans['Selected_Frames'][i])
            mylist = df_pos_scans['Selected_Frames'][i].split('-')
            mylist[0]=mylist[0][1:]
            mylist[1]=mylist[1][:-1]
            mylist = [int(i) for i in mylist]
            print(type(mylist[0]))
            start_i = mylist[0]
            end_i = mylist[1]+1
            for slice_num in range(start_i,end_i):
                slice_name = name + '_' + df_pos_scans['Laterality'][i] + '_' + df_pos_scans['View'][i]+'_'+str(slice_num)+'.png'
                if slice == slice_name:
                    pos.append(slice)
    else:
        for slice in train_untagged_list:
            if name in slice:
                to_check.append(name)

print('to check:')
print(to_check)
df = pd.concat([pd.Series(pos)], ignore_index=True, axis=1)
df.columns =['test']
df.to_excel('test.xlsx',index = False)