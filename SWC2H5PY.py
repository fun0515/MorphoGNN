import h5py
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D

def ReadH5py(dir,normalization=True):
    f = h5py.File(dir,'r')
    # print(f.keys())
    data = f['data'][:]
    if normalization:
        data = Normalization(data)
    label = f['label'][:]
    f.close()
    return data,label

def Normalization(data):
    data_normalized = np.zeros(data.shape)
    for i in range(0,data.shape[0]):
        temp = data[i]
        origin = np.zeros((1,3))
        origin[0][0] = (temp[:,0].max() + temp[:,0].min()) / 2
        origin[0][1] = (temp[:,1].max() + temp[:,1].min()) / 2
        origin[0][2] = (temp[:,2].max() + temp[:,2].min()) / 2
        temp[:,0] = temp[:,0] - origin[0][0]
        temp[:,1] = temp[:,1] - origin[0][1]
        temp[:,2] = temp[:,2] - origin[0][2]
        if temp[:,0].max()>1:
            temp[:,0] = temp[:,0] / temp[:,0].max()
        if temp[:, 1].max() > 1:
            temp[:,1] = temp[:,1] / temp[:,1].max()
        if temp[:, 2].max() > 1:
            temp[:,2] = temp[:,2] / temp[:,2].max()
        data_normalized[i] = temp
    return data_normalized

def WriteH5py(dir,data,label):
    f = h5py.File(dir,'w')
    f['data'] = data
    f['label'] = label


def ReadSWC(dir,thresold,CLIP=False,Padding=True):
    data = []
    with open(dir,'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if line[0] == '#'or line[0] == '\n':
                continue
            _,_,x,y,z,_,_ = [float(i) for i in line.split()]
            data.append([x,y,z])
        f.close()
        if Padding:
            while len(data)<thresold:data.append([0,0,0])
        if CLIP:
            length = math.floor(len(data) / thresold)
            data = np.array(data[0:(length*thresold)])
        else:data = np.array(data)
    return data

def GenerateH5py(dir_list,thresold):
    label10 = {'pyramidal':0,'aspiny':1,'cholinergic':2,'ganglion':3,'basket':4,'fast-spiking':5,'sensory':6,'neurogliaform':7,'martinotti':8,'mitral':9}
    label7 = {'amacrine':0,'aspiny':1,'basket':2,'bipolar':3,'pyramidal':4,'spiny':5,'stellate':6}
    i = 0
    for filename in os.listdir(dir_list):
        if filename.split('.')[-1] != 'swc':continue
        print(dir_list.split('/')[-1],'/',filename,' ',i)
        if ReadSWC(dir_list+'/'+filename,thresold).shape[0]<thresold:
            continue
        if i == 0:
            datas = ReadSWC(dir_list+'/'+filename,thresold)
            i = i + 1
            continue
        datas = np.concatenate((datas,ReadSWC(dir_list+'/'+filename,thresold)))
        i = i + 1
    if i==0:
        return 'continue','continue'
    datas = datas.reshape(-1,thresold,3)
    labels = np.ones((datas.shape[0], 1))*int(label7[dir_list.split('/')[-1]])
    return datas,labels

def GenerateNeuronDataset(neuron_list,thresold,proportion):
    for i,neuron_type in enumerate(os.listdir(neuron_list)):
        if i == 0:
            datas,labels = GenerateH5py(neuron_list+'/'+neuron_type,thresold)
            continue
        data,label = GenerateH5py(neuron_list+'/'+neuron_type,thresold)
        if data == 'continue':
            continue
        datas = np.concatenate((datas,data))
        labels = np.concatenate((labels,label))
    print(datas.shape,' ',labels.shape)
    state = np.random.get_state()
    np.random.shuffle(datas)
    np.random.set_state(state)
    np.random.shuffle(labels)
    datas = datas.astype(np.float32)
    labels = labels.astype(np.uint8)
    WriteH5py(dir = r'./TrainDatasets_6000.h5',data=datas[0:math.ceil(proportion*datas.shape[0])],
              label=labels[0:math.ceil(proportion*labels.shape[0])])
    WriteH5py(dir=r'./TestDatasets_6000.h5', data=datas[math.ceil(proportion * datas.shape[0]):-1],
              label=labels[math.ceil(proportion * labels.shape[0]):-1])


def VisualizeH5py(dir):
    datas,labels = ReadH5py(dir,normalization=False)
    datas2,labels2 = ReadH5py(dir,normalization=True)
    fig = plt.figure(dpi=180)
    ax1 = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(0,200):
        data = datas[i]
        data2 = datas2[i]
        ax1.cla()
        ax1.scatter(data[:,0],data[:,1],data[:,2],c="b", marker=".", s=15, linewidths=0, alpha=1, cmap="spectral")
        ax2.cla()
        ax2.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c="r", marker=".", s=15, linewidths=0, alpha=1,cmap="spectral")
        plt.pause(0.5)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate morphological dataset')
    parser.add_argument('--swc_dir',type=str,default='./neuron7',help='file path of .swc files')
    args = parser.parse_args()
    # VisualizeH5py(r'DataSets/neuron7/TrainDatasets_6000.h5')
    GenerateNeuronDataset(neuron_list=args.swc_dir,thresold=6000,proportion=0.7)

