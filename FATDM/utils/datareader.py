import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .data_utils import read_metadata, convert_age_2_num, get_age_range, convert_gender_2_num
from sklearn.model_selection import train_test_split


class ConcatDomainRotatedXrayDataset(Dataset):
    def __init__(self, metadata_file_list, data_dir, dataset, test_domain, lb_val, g_val):
        assert dataset in ['train', 'val', 'test']
        metadata_list = {}
        domain_list = ['0', '15', '30', '45', '60']
        if g_val is not None:
            age_range = get_age_range(g_val)
        for d, metadata_file in zip(domain_list, metadata_file_list):
            mtdt = read_metadata(metadata_file)
            mtdt['label'] = mtdt['label'].astype(np.float32)
            if lb_val is not None and g_val is not None:
                mtdt = mtdt[(mtdt['label'] == lb_val) & (mtdt['anchor_age'].isin(age_range))]
            elif lb_val is not None:
                mtdt = mtdt[mtdt['label'] == lb_val]
            else:
                pass
            metadata_list[d] = mtdt
        if dataset == 'test':
            self.metadata = metadata_list[test_domain].to_numpy()
        else:
            train_metadata_list = []
            val_metadata_list = []
            for d, mtdt in metadata_list.items():
                if d != test_domain:
                    idx = list(range(len(mtdt)))
                    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
                    train_metadata_list.append(mtdt.iloc[list(train_idx)])
                    val_metadata_list.append(mtdt.iloc[list(val_idx)])
            if dataset == 'train':
                self.metadata = pd.concat(train_metadata_list).to_numpy()
            else:
                self.metadata = pd.concat(val_metadata_list).to_numpy()
            train_domain = sorted(list(set(self.metadata[:, 7])))
            self.train_domain_dict = dict(zip(train_domain, list(range(len(train_domain)))))
        self.data_dir = data_dir
        self.dataset = dataset

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        d_lb = self.metadata[idx, 7]
        img_path = self.data_dir + '/' + str(d_lb) + '/' + self.metadata[idx, 0] + '.npy'
        img = np.load(img_path)[np.newaxis, :]
        img = np.asarray(img, dtype=np.float32)
        age = convert_age_2_num(self.metadata[idx, 4])
        lb = self.metadata[idx, 6]
        if self.dataset in ['train', 'val']:
            d_lb = self.train_domain_dict[d_lb]
            return {'img': img, 'age': age, 'lb': lb, 'd_lb': d_lb}
        else:
            return {'img': img, 'age': age, 'lb': lb}


class MultiRotatedXrayDataset(Dataset):
    def __init__(self, metadata_file_list, data_dir, dataset, test_domain):
        assert dataset in ['train', 'val', 'test']
        metadata_list = {}
        domain_list = ['0', '15', '30', '45', '60']
        for d, metadata_file in zip(domain_list, metadata_file_list):
            mtdt = read_metadata(metadata_file)
            mtdt['label'] = mtdt['label'].astype(np.float32)
            metadata_list[d] = mtdt
        if dataset == 'test':
            self.metadata = metadata_list[test_domain].to_numpy()
        else:
            train_metadata_list = []
            val_metadata_list = []
            for d, mtdt in sorted(metadata_list.items()):
                if d != test_domain:
                    idx = list(range(len(mtdt)))
                    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
                    train_metadata_list.append(mtdt.iloc[list(train_idx)].to_numpy())
                    val_metadata_list.append(mtdt.iloc[list(val_idx)].to_numpy())
            if dataset == 'train':
                self.metadata = train_metadata_list
            else:
                self.metadata = val_metadata_list
        self.data_dir = data_dir
        self.length_list = [len(dt) for dt in self.metadata]
        self.dataset = dataset

    def __len__(self):
        if self.dataset in ['train', 'val']:
            return np.max([self.length_list])
        else:
            return len(self.metadata)

    def shuffle_metadata(self):
        for mtdt in self.metadata:
            np.random.shuffle(mtdt)

    def __getitem__(self, idx):
        if self.dataset in ['train', 'val']:
            idx_list = [idx % i for i in self.length_list]
            metadata_list = [mtdt[idx] for mtdt, idx in zip(self.metadata, idx_list)]
            img_path_list = [self.data_dir + '/' + str(mtdt[7]) + '/' + mtdt[0] + '.npy' for mtdt in metadata_list]
            img_list = [np.load(img_path)[np.newaxis, :] for img_path in img_path_list]
            age_list = [convert_age_2_num(mtdt[4]) for mtdt in metadata_list]
            lb_list = [mtdt[6] for mtdt in metadata_list]
            domain_lb_list = np.arange(len(self.length_list), dtype=np.float32)
            return {'img': np.array(img_list), 'age': np.array(age_list), 'lb': np.array(lb_list),
                    'd_lb': np.array(domain_lb_list)}
        else:
            img_path = self.data_dir + '/' + str(self.metadata[idx, 7]) + '/' + self.metadata[idx, 0] + '.npy'
            img = np.load(img_path)[np.newaxis, :]
            age = convert_age_2_num(self.metadata[idx, 4])
            lb = self.metadata[idx, 6]
            return {'img': img, 'age': age, 'lb': lb}


class ConcatDomainAgeXrayDataset(Dataset):
    def __init__(self, metadata_file_list, data_dir, dataset, test_domain, lb_val, g_val):
        assert dataset in ['train', 'val', 'test']
        metadata_list = {}
        domain_list = ['0_40', '40_60', '60_80', '80_100']
        gender_map = {0: 'F', 1: 'M'}
        for d, metadata_file in zip(domain_list, metadata_file_list):
            mtdt = read_metadata(metadata_file)
            mtdt['label'] = mtdt['label'].astype(np.float32)
            if lb_val is not None and g_val is not None:
                mtdt = mtdt[(mtdt['label'] == lb_val) & (mtdt['gender'] == gender_map[g_val])]
            elif lb_val is not None:
                mtdt = mtdt[mtdt['label'] == lb_val]
            else:
                pass
            metadata_list[d] = mtdt
        if dataset == 'test':
            self.metadata = metadata_list[test_domain].to_numpy()
        else:
            train_metadata_list = []
            val_metadata_list = []
            for d, mtdt in metadata_list.items():
                if d != test_domain:
                    idx = list(range(len(mtdt)))
                    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
                    train_metadata_list.append(mtdt.iloc[list(train_idx)])
                    val_metadata_list.append(mtdt.iloc[list(val_idx)])
            if dataset == 'train':
                self.metadata = pd.concat(train_metadata_list).to_numpy()
            else:
                self.metadata = pd.concat(val_metadata_list).to_numpy()
            train_domain = sorted(list(set(self.metadata[:, 7])))
            self.train_domain_dict = dict(zip(train_domain, list(range(len(train_domain)))))
        self.data_dir = data_dir
        self.dataset = dataset

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        d_lb = self.metadata[idx, 7]
        img_path = self.data_dir + '/' + str(d_lb) + '/' + self.metadata[idx, 0] + '.npy'
        img = np.load(img_path)[np.newaxis, :]
        img = np.asarray(img, dtype=np.float32)
        gender = convert_gender_2_num(self.metadata[idx, 3])
        lb = self.metadata[idx, 6]
        if self.dataset in ['train', 'val']:
            d_lb = self.train_domain_dict[d_lb]
            return {'img': img, 'gender': gender, 'lb': lb, 'd_lb': d_lb}
        else:
            return {'img': img, 'gender': gender, 'lb': lb}


class MultiAgeXrayDataset(Dataset):
    def __init__(self, metadata_file_list, data_dir, dataset, test_domain):
        assert dataset in ['train', 'val', 'test']
        metadata_list = {}
        domain_list = ['0_40', '40_60', '60_80', '80_100']
        for d, metadata_file in zip(domain_list, metadata_file_list):
            mtdt = read_metadata(metadata_file)
            mtdt['label'] = mtdt['label'].astype(np.float32)
            metadata_list[d] = mtdt
        if dataset == 'test':
            self.metadata = metadata_list[test_domain].to_numpy()
        else:
            train_metadata_list = []
            val_metadata_list = []
            for d, mtdt in sorted(metadata_list.items()):
                if d != test_domain:
                    idx = list(range(len(mtdt)))
                    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
                    train_metadata_list.append(mtdt.iloc[list(train_idx)].to_numpy())
                    val_metadata_list.append(mtdt.iloc[list(val_idx)].to_numpy())
            if dataset == 'train':
                self.metadata = train_metadata_list
            else:
                self.metadata = val_metadata_list
        self.data_dir = data_dir
        self.length_list = [len(dt) for dt in self.metadata]
        self.dataset = dataset

    def __len__(self):
        if self.dataset in ['train', 'val']:
            return np.max([self.length_list])
        else:
            return len(self.metadata)

    def shuffle_metadata(self):
        for mtdt in self.metadata:
            np.random.shuffle(mtdt)

    def __getitem__(self, idx):
        if self.dataset in ['train', 'val']:
            idx_list = [idx % i for i in self.length_list]
            metadata_list = [mtdt[idx] for mtdt, idx in zip(self.metadata, idx_list)]
            img_path_list = [self.data_dir + '/' + str(mtdt[7]) + '/' + mtdt[0] + '.npy' for mtdt in metadata_list]
            img_list = [np.load(img_path)[np.newaxis, :] for img_path in img_path_list]
            gender_list = [convert_gender_2_num(mtdt[3]) for mtdt in metadata_list]
            lb_list = [mtdt[6] for mtdt in metadata_list]
            domain_lb_list = np.arange(len(self.length_list), dtype=np.float32)
            return {'img': np.array(img_list), 'gender': np.array(gender_list), 'lb': np.array(lb_list),
                    'd_lb': np.array(domain_lb_list)}
        else:
            img_path = self.data_dir + '/' + str(self.metadata[idx, 7]) + '/' + self.metadata[idx, 0] + '.npy'
            img = np.load(img_path)[np.newaxis, :]
            gender = convert_gender_2_num(self.metadata[idx, 3])
            lb = self.metadata[idx, 6]
            return {'img': img, 'gender': gender, 'lb': lb}
