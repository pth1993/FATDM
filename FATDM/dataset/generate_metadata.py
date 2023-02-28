import pandas as pd
import random
from collections import Counter


def load_data_2_df(input_file, columns=None):
    data = pd.read_csv(input_file)
    if columns:
        data = data[columns]
    return data


def remove_duplicate_entries(df):
    data_dict = dict()
    data_numpy = df.to_numpy()
    for row in data_numpy:
        if row[0] not in data_dict:
            data_dict[row[0]] = [row[1]]
        else:
            data_dict[row[0]].append(row[1])
    for k, v in data_dict.items():
        if len(set(v)) > 1 and len(v) > 2:
            print(k, v)
    return data_dict


if __name__ == '__main__':
    ethic = ['HISPANIC/LATINO', 'ASIAN', 'BLACK/AFRICAN AMERICAN', 'AMERICAN INDIAN/ALASKA NATIVE', 'WHITE', 'OTHER']
    disease_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                    'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
                    'Support Devices']
    for disease in disease_list:
        # label_info
        label = load_data_2_df('label-chexpert.csv', ['subject_id', 'study_id', 'No Finding', disease])
        label = label[(label[disease].isin([1.0, 0.0]))]
        label['label'] = (label[disease] == 1.0) * 1.0
        label = label[['subject_id', 'study_id', 'label']]

        # patient_info
        patient = load_data_2_df('patients.csv', ['subject_id', 'gender', 'anchor_age'])
        admission = load_data_2_df('admissions.csv', ['subject_id', 'ethnicity'])
        admission = admission[admission['ethnicity'].isin(ethic)]
        admission_filter = admission.drop_duplicates()
        unique_idx = admission_filter[admission_filter.duplicated('subject_id', keep=False)]['subject_id']
        admission_filter = admission_filter[~admission_filter['subject_id'].isin(unique_idx)]
        patient_info = pd.merge(patient, admission_filter, how="inner", on=['subject_id'])
        patient_label = pd.merge(patient_info, label, how="inner", on=['subject_id'])

        # image_info
        image = load_data_2_df('mimic-cxr-2.0.0-metadata.csv', ['dicom_id', 'subject_id', 'study_id'])
        metadata = pd.merge(image, patient_label, how="inner", on=['subject_id', 'study_id'])

        cnt = [0, 0, 0, 0]
        age_list = metadata['anchor_age'].tolist()
        for y_neg in age_list:
            if y_neg < 40:
                cnt[0] += 1
            elif y_neg < 60:
                cnt[1] += 1
            elif y_neg < 80:
                cnt[2] += 1
            else:
                cnt[3] += 1
        print('age', cnt)
        gender_list = metadata['gender'].tolist()
        print('gender', Counter(gender_list))
        label_list = metadata['label'].tolist()
        print('label', Counter(label_list))

        y_neg = metadata[(metadata['label'] == 0.0) & (0 <= metadata['anchor_age']) & (metadata['anchor_age'] < 40)]
        y_pos = metadata[(metadata['label'] == 1.0) & (0 <= metadata['anchor_age']) & (metadata['anchor_age'] < 40)]
        a_female = metadata[(metadata['gender'] == 'F') & (0 <= metadata['anchor_age']) & (metadata['anchor_age'] < 40)]
        a_male = metadata[(metadata['gender'] == 'M') & (0 <= metadata['anchor_age']) & (metadata['anchor_age'] < 40)]
        print('0 < age group < 40 (P(Y)): %.4f' % (len(y_neg) / (len(y_neg) + len(y_pos))))
        print('0 < age group < 40 (P(A)): %.4f' % (len(a_female) / (len(a_female) + len(a_male))))

        y_neg = metadata[(metadata['label'] == 0.0) & (40 <= metadata['anchor_age']) & (metadata['anchor_age'] < 60)]
        y_pos = metadata[(metadata['label'] == 1.0) & (40 <= metadata['anchor_age']) & (metadata['anchor_age'] < 60)]
        a_female = metadata[(metadata['gender'] == 'F') & (40 <= metadata['anchor_age']) & (metadata['anchor_age'] < 60)]
        a_male = metadata[(metadata['gender'] == 'M') & (40 <= metadata['anchor_age']) & (metadata['anchor_age'] < 60)]
        print('40 < age group < 60 (P(Y)): %.4f' % (len(y_neg) / (len(y_neg) + len(y_pos))))
        print('40 < age group < 60 (P(A)): %.4f' % (len(a_female) / (len(a_female) + len(a_male))))

        y_neg = metadata[(metadata['label'] == 0.0) & (60 <= metadata['anchor_age']) & (metadata['anchor_age'] < 80)]
        y_pos = metadata[(metadata['label'] == 1.0) & (60 <= metadata['anchor_age']) & (metadata['anchor_age'] < 80)]
        a_female = metadata[(metadata['gender'] == 'F') & (60 <= metadata['anchor_age']) & (metadata['anchor_age'] < 80)]
        a_male = metadata[(metadata['gender'] == 'M') & (60 <= metadata['anchor_age']) & (metadata['anchor_age'] < 80)]
        print('60 < age group < 80 (P(Y)): %.4f' % (len(y_neg) / (len(y_neg) + len(y_pos))))
        print('60 < age group < 80 (P(A)): %.4f' % (len(a_female) / (len(a_female) + len(a_male))))

        y_neg = metadata[(metadata['label'] == 0.0) & (80 <= metadata['anchor_age']) & (metadata['anchor_age'] < 100)]
        y_pos = metadata[(metadata['label'] == 1.0) & (80 <= metadata['anchor_age']) & (metadata['anchor_age'] < 100)]
        a_female = metadata[(metadata['gender'] == 'F') & (80 <= metadata['anchor_age']) & (metadata['anchor_age'] < 100)]
        a_male = metadata[(metadata['gender'] == 'M') & (80 <= metadata['anchor_age']) & (metadata['anchor_age'] < 100)]
        print('80 < age group < 100 (P(Y)): %.4f' % (len(y_neg) / (len(y_neg) + len(y_pos))))
        print('80 < age group < 100 (P(A)): %.4f' % (len(a_female) / (len(a_female) + len(a_male))))

        domain_list = ['0_40', '40_60', '60_80', '80_100']
        for i, (d, a) in enumerate(zip(domain_list, [[0, 40], [40, 60], [60, 80], [80, 100]])):
            mtdt = metadata[(a[0] <= metadata['anchor_age']) & (metadata['anchor_age'] < a[1])].reset_index(drop=True)
            idx_list = list(range(len(mtdt)))
            idx = random.Random(i).sample(idx_list, 5000)
            mtdt.loc[idx, 'age_group'] = d
            age_metadata = mtdt.loc[idx]
            age_metadata.to_csv('metadata/age_%s/%s_%s.csv' % (disease.lower(), disease, d), index=False)
