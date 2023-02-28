import pandas as pd


def read_metadata(input_file):
    metadata = pd.read_csv(input_file)
    return metadata


def split_xray_data_by_domain(data, domain):
    domain_list = data[domain].unique()
    data_list = dict()
    for d in domain_list:
        data_list[d] = data[data[domain] == d]
    return data_list


def convert_age_2_num(age):
    if age < 40:
        return 0
    elif age < 60:
        return 1
    elif age < 80:
        return 2
    else:
        return 3


def convert_gender_2_num(gender):
    if gender == 'F':
        return 0
    elif gender == 'M':
        return 1
    else:
        raise ValueError('Wrong gender')


def get_age_range(g_val):
    if g_val == 0:
        age_range = list(range(40))
    elif g_val == 1:
        age_range = list(range(40, 60))
    elif g_val == 2:
        age_range = list(range(60, 80))
    else:
        age_range = list(range(80, 120))
    return age_range
