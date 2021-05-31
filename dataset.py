import os
from SWC2H5PY import ReadH5py
import torch.utils.data as data
import numpy as np
import h5py
import logging  # 引入logging模块
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

train_dir = r'./DataSets/train_6000.h5py'
test_dir = r'./DataSets/test_6000.h5py'

class DataSet(data.Dataset):
    def __init__(self, train=True):
        if train:
            data,label = ReadH5py(train_dir,normalization=True)
        else:
            data,label = ReadH5py(test_dir,normalization=True)

        logging.info('data: {}'.format(data.shape))
        logging.info('label: {}'.format(label.shape))

        self.point_cloud = data
        self.label = label

    def __getitem__(self, item):
        return self.point_cloud[item], self.label[item]

    def __len__(self):
        return self.label.shape[0]




if __name__ == '__main__':
    train = DataSet(train=True)
    test = DataSet(train=False)

    data, label = train[0]
    print('type: {}'.format(type(data)))
    print(len(train))
    print('type: {}'.format(type(label)))
    print(label.shape)