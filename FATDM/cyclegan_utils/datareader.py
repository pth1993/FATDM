import numpy as np
from torch.utils.data import Dataset
from data_utils import read_metadata, convert_gender_2_num
from sklearn.model_selection import train_test_split
import pandas as pd


class SingleDomainAgeXrayDataset(Dataset):
    def __init__(self, metadata_file, data_dir, domain, lb_val, g_val):
        gender_map = {0: 'F', 1: 'M'}
        mtdt = read_metadata(metadata_file)
        mtdt['label'] = mtdt['label'].astype(np.float32)
        if lb_val is not None and g_val is not None:
            mtdt = mtdt[(mtdt['label'] == lb_val) & (mtdt['gender'] == gender_map[g_val])]
        elif lb_val is not None:
            mtdt = mtdt[mtdt['label'] == lb_val]
        else:
            pass
        idx = list(range(len(mtdt)))
        train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
        self.metadata = mtdt.iloc[list(train_idx)].to_numpy()
        self.data_dir = data_dir
        self.domain = domain

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        d_lb = self.domain
        img_path = self.data_dir + '/' + str(d_lb) + '/' + self.metadata[idx, 0] + '.npy'
        img = np.load(img_path)[np.newaxis, :]
        img = np.asarray(img, dtype=np.float32)
        lb = self.metadata[idx, 6]
        return {'img': img, 'lb': lb}


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