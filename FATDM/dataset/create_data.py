import csv
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm


def get_img_url(row):
    url = 'p' + row[1][:2] + '/p' + row[1] + '/s' + row[2] + '/' + row[0] + '.jpg'
    return url


def generate_url_data(input_file, output_file):
    url_list = []
    with open(input_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            url = get_img_url(row)
            url_list.append(url)
    url_list_string = '\n'.join(url_list) + '\n'
    with open(output_file, 'w') as f:
        f.write(url_list_string)
    return url_list


def copy_data(url_list, src_prefix, target_prefix):
    for i, url in enumerate(url_list):
        shutil.copy2(src_prefix + url, target_prefix)


def create_numpy_data():
    with open('metadata/Edema.csv') as f1:
        f1.readline()
        for i, line in enumerate(tqdm(f1)):
            line = line.strip().split(',')
            img_path = 'data/edema/' + line[0] + '.jpg'
            img = Image.open(img_path).resize((256, 256))
            img = np.array(img, np.float32)
            img = (img - img.mean()) / (img.std() + 1e-8)
            with open('data/edema_numpy/' + line[0] + '.npy', 'wb') as f2:
                np.save(f2, img)


def create_rotated_numpy_data():
    def process_img(img, angle):
        img = np.array(img, np.float32)
        img = img / 255
        img = (img - 0.5) / 0.5
        with open('data/rotated_edema_numpy_1/%s/' % angle + line[0] + '.npy', 'wb') as f2:
            np.save(f2, img)

    with open('metadata/Edema_no_finding.csv') as f1:
        f1.readline()
        for i, line in enumerate(tqdm(f1)):
            line = line.strip().split(',')
            img_path = 'data/edema/' + line[0] + '.jpg'
            img_0 = Image.open(img_path).resize((256, 256))
            img_15 = img_0.rotate(15)
            img_30 = img_0.rotate(30)
            img_45 = img_0.rotate(45)
            img_60 = img_0.rotate(60)
            process_img(img_0, '0')
            process_img(img_15, '15')
            process_img(img_30, '30')
            process_img(img_45, '45')
            process_img(img_60, '60')


def create_age_numpy_data(mtdt_file, domain):
    def process_img(img, img_name, domain):
        img = np.array(img, np.float32)
        img = img / 255
        img = (img - 0.5) / 0.5
        with open('data/age_cardiomegaly_numpy/%s/' % domain + img_name + '.npy', 'wb') as f1:
            np.save(f1, img)

    with open(mtdt_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in tqdm(reader):
            img_url = get_img_url(row)
            img_name = row[0]
            src_prefix = '/fs/project/PAS1536/hoang/datasets/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
            img = Image.open(src_prefix + img_url).resize((256, 256))
            process_img(img, img_name, domain)


if __name__ == '__main__':
    mtdt_file_list = ['Cardiomegaly_0_40.csv', 'Cardiomegaly_40_60.csv', 'Cardiomegaly_60_80.csv',
                      'Cardiomegaly_80_100.csv']
    domain_list = ['0_40', '40_60', '60_80', '80_100']
    for mtdt_file, domain in zip(mtdt_file_list, domain_list):
        create_age_numpy_data('metadata/age_cardiomegaly/%s' % mtdt_file, domain=domain)
