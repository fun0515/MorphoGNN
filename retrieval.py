import torch
import argparse
import os
import numpy as np
import time
import argparse
import random
import h5py
from torch.autograd import Variable
from collections import OrderedDict
from SWC2H5PY import ReadSWC,Normalization
import matplotlib.pyplot as plt
from sklearn import manifold,datasets
from MorphoGNN import MorphoGNN

def load_model(name='morpho',model_path='./MorphoGNN.t7'):
    device = torch.device('cuda')
    if name == 'morpho':
        model = MorphoGNN().to(device)
    else:
        raise Exception("Not implemented")
    model.load_state_dict(torch.load(model_path),False)
    return model

def ExtractFeature(model_path,neuron_list):
    model = load_model('morpho',model_path)
    model.eval()
    NeuronVectorDatabase = {}
    with torch.no_grad():
        for i, neuron_type in enumerate(os.listdir(neuron_list)):
            for j, swc in enumerate(os.listdir(neuron_list + '/' + neuron_type)):
                if swc.split('.')[-1] != 'swc': continue
                input = ReadSWC(neuron_list + '/' + neuron_type + '/' + swc, thresold=6000, CLIP=False,Padding=True)
                input = input.reshape(((1,) + input.shape))
                input = Normalization(input)
                input = torch.tensor(input)
                input = input.permute(0, 2, 1)
                input = input.type(torch.FloatTensor)
                input = input.to('cuda')
                if input.shape[-1] >= 28000 or input.shape[-1] <= 100:
                    continue
                print(i, ' ', j, '', input.shape[-1])
                out = model(input)
                feature = model.feature
                feature = feature.cpu().numpy().squeeze()
                NeuronVectorDatabase[neuron_type + '/' + swc] = feature
    np.save(r'./database.npy', NeuronVectorDatabase)

def CaculateEucliDist(a,b):
    return np.sqrt(sum(np.power((a - b), 2)))

def CaculateCosineSimilarity(a,b):
    num = a.dot(b.T)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return num / denom

def CaculateManhattanDist(a,b):
    return sum(abs(x - y) for x, y in zip(a, b))

def ReturnKey(dict,value):
    '''input value and return key'''
    for k, v in dict.items():
        if all(v==value):
            return k

def CaculateAcc(query,result):
    query_type = query[0:query.find('/')]
    result_type = []
    for swc in result:
        result_type.append(swc[0:swc.find('/')])
    return result_type.count(query_type)/len(result_type),query_type

def QueryTest(database,i):
    '''{'swc_path':vector}'''
    database = np.load(database, allow_pickle=True).item()
    random.seed(3)
    #query_swc = random.sample(database.keys(), len(database))[i]
    query_swc = 'spiny/Scnn1a-Tg3-Cre-Ai14-475124505.CNG.swc'
    print(query_swc)
    query_vec = database[query_swc]
    del database[query_swc]
    value = [value for value in database.values()]

    dist = [CaculateCosineSimilarity(query_vec, val) for val in value]
    dist_value = dict(zip(dist, value))
    dist_value = sorted(dist_value.items(), key=lambda x: x[0], reverse=True)  # 按dist排序

    result = []
    for i in range(0, 10):
        result.append(ReturnKey(database, dist_value[i][-1]))
        print(result[i],' ',dist_value[i][0])
    VisualizeQueryResult(query_swc,result)
    return CaculateAcc(query_swc, result)

def QueryTests(npy):
    acc = []
    class_acc = {}
    sum_acc = 0
    for i in range(0, len(np.load(npy, allow_pickle=True).item())):
        print(i)
        instance_acc,type = QueryTest(npy,i)
        acc.append(instance_acc)
        if type not in class_acc:
            temp = []
            temp.append(instance_acc)
            class_acc[type] = temp
        else: class_acc[type].append(instance_acc)
    for ty in class_acc.keys():
        sum_acc = sum_acc + sum(class_acc[ty])/len(class_acc[ty])
    print('acc: ',sum(acc) / len(acc))
    print('avg acc: ',sum_acc/len(class_acc))

def ReturnLabel(database):
    label_list = []
    label10 = {'pyramidal': 0, 'aspiny': 1, 'cholinergic': 2, 'ganglion': 3, 'basket': 4, 'fast-spiking': 5, 'sensory': 6,
             'neurogliaform': 7, 'martinotti': 8, 'mitral': 9}
    label7 = {'amacrine': 0, 'aspiny': 1, 'basket': 2, 'bipolar': 3, 'pyramidal': 4, 'spiny': 5, 'stellate': 6}
    for swc in database.keys():
        label_list.append(label7[swc[0:swc.find('/')]])
    return np.array(label_list)


def Tsne(database):
    label7 = ['Amacrine', 'Aspiny', 'Basket', 'Bipolar', 'Pyramidal', 'Spiny', 'Stellate']
    database = np.load(database, allow_pickle=True).item()
    feature = [value for value in database.values()]
    feature = np.array(feature)
    label = ReturnLabel(database)
    print(feature.shape)
    print(label.shape)
    tsne = manifold.TSNE(n_components=2,init='pca',random_state=13)
    X_tsne = tsne.fit_transform(feature)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(6, 6))
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set2(label[i]),label = label7[label[i]])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc='upper right',shadow=False,frameon=False,handletextpad=0.2,fontsize=10) #fontsize=14

    plt.title('Feature',size=20)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def VisualizeQueryResult(query,dir_list):
    fig = plt.figure(dpi=300)
    query_data = ReadSWC('C:/Users\90674\Desktop\siqi\siqi/neuron7'+'/'
                       +query,thresold=6000,CLIP=False,Padding=True)
    ax1 = fig.add_subplot(3,5,3, projection='3d')
    ax1.scatter(query_data[:, 0], query_data[:, 1], query_data[:, 2], c="b", marker=".", s=1, linewidths=0, alpha=1, cmap="spectral")
    ax1.axis('off')
    for i in range(len(dir_list)):
        data = ReadSWC('./neuron7'+'/'
                       +dir_list[i],thresold=6000,CLIP=False,Padding=True)
        ax = fig.add_subplot(3,5,i+6, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="b", marker=".", s=1, linewidths=0, alpha=1, cmap="spectral")
        ax.axis('off')
    plt.show()

def ReturnClassFeature(npy,type):
    database = np.load(npy, allow_pickle=True).item()
    result = {}
    for key in database.keys():
        if key[0:key.find('/')] == type:
            result[key] = database[key]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='neuron indexing')
    parser.add_argument('--task', type=str, help='choose your task',choices=['ExtractFeature','QueryTest','Visualize'])
    parser.add_argument('--query_times',type=int)
    parser.add_argument('--swc_dir', type=str,default='./neuron7')
    parser.add_argument('--model_path', type=str, default='./MorphoGNN.t7')
    args = parser.parse_args()
    if args.task == 'ExtractFeature':
        ExtractFeature(args.model_path,args.swc_dir)
    elif args.task == 'QueryTest':
        QueryTest('database.npy', args.query_times)
    elif args.task == 'Visualize':
        Tsne('database.npy')






