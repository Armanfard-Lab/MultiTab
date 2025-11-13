import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from torch.utils.data import Dataset
import h5py

class AliExpressDataset_Legacy(Dataset):
    """
    AliExpress Dataset
    This is a dataset gathered from real-world traffic logs of the search system in AliExpress
    Reference:
        https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690
        Li, Pengcheng, et al. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. CIKM 2020.
    """

    def __init__(self, tasks, dataset_path, train, seperate_ft_types=False, indices=None):
        self.tasks = tasks
        self.seperate_ft_types = seperate_ft_types
        self.dataset_path = dataset_path + 'train.csv' if train else dataset_path + 'test.csv'

        df = pd.read_csv(self.dataset_path)

        # Drop columns with only 1 unique value
        df.drop(columns=['categorical_1', 'numerical_58', 'numerical_59', 'numerical_60'], inplace=True)

        data = df.to_numpy()[:, 1:]
        if indices is not None:
            data = data[indices]
        self.categorical_data = data[:, :15]
        self.numerical_data = data[:, 15: -2]
        self.labels = data[:, -2:]

        # Ensure categorical variables are sequentially encoded starting at index 0
        self.categorical_data = self._encode_categories(self.categorical_data)

        self.numerical_num = self.numerical_data.shape[1]
        cat_dims = np.max(self.categorical_data, axis=0) + 1
        num_dims = np.ones(self.numerical_num)
        self.field_dims = np.append(cat_dims, num_dims)

    def __len__(self):
        return self.labels.shape[0]
    
    def _encode_categories(self, categorical_data):
        encoder = OrdinalEncoder()
        return encoder.fit_transform(categorical_data)

    def __getitem__(self, index):
        categorical = torch.from_numpy(self.categorical_data[index])
        continuous = torch.from_numpy(self.numerical_data[index])
        if self.seperate_ft_types:
            features = {
                'categorical': categorical.to(torch.long),
                'continuous': continuous.to(torch.float32)
            }
        else:
            features = torch.cat((categorical, continuous)).to(torch.float32)
        targets = torch.from_numpy(self.labels[index]).to(torch.float32)
        
        sample = {
            'features': features,
            'targets': {self.tasks[i]: targets[i] for i in range(len(self.tasks))}
        }

        return sample

class AliExpressDataset(Dataset):
    def __init__(self, h5_path, split, seperate_ft_types=False):
        self.h5_path = h5_path
        self.split = split
        self.seperate_ft_types = seperate_ft_types

        # Open and load into RAM as numpy arrays
        with h5py.File(self.h5_path, 'r') as file:
            self.x_cat = torch.from_numpy(file[f'{self.split}/features_categorical'][:]).float()
            self.x_num = torch.from_numpy(file[f'{self.split}/features_numerical'][:]).float()
            self.y_click = torch.from_numpy(file[f'{self.split}/click'][:]).float()
            self.y_conversion = torch.from_numpy(file[f'{self.split}/conversion'][:]).float()

        self.length = self.y_click.shape[0]

        # Compute field dims (can still use NumPy)
        cat_dims = torch.max(self.x_cat, dim=0).values + 1
        num_dims = torch.ones(self.x_num.shape[1], dtype=torch.long)
        self.field_dims = torch.cat([cat_dims, num_dims])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_cat = self.x_cat[idx]
        x_num = self.x_num[idx]
        y_click = self.y_click[idx]
        y_conversion = self.y_conversion[idx]

        if self.seperate_ft_types:
            features = {
                'categorical': x_cat.long(),
                'continuous': x_num
            }
        else:
            features = torch.cat((x_cat, x_num), dim=0)  # ensure float32

        targets = {
            'click': y_click,
            'conversion': y_conversion
        }

        return {
            'features': features,
            'targets': targets
        }
