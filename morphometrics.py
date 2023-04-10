import neurom as nm
import numpy as np
import logging
import time
import h5py
import argparse
from MorphoGNN import *

import math
import os

def WriteH5py(dir,data,label):
    f = h5py.File(dir,'w')
    f['data'] = data
    f['label'] = label

def     ReturnFeatures(neuron):
    features = []
    features = []
    features.append(nm.get('n_sections', neuron))
    features.append(nm.get('n_leaves', neuron))
    features.append(max(nm.get('section_branch_orders', neuron)))
    features.append(nm.get('max_radial_distance', neuron))
    features.append(sum(nm.get('section_lengths', neuron)))  # equal to 'total length'
    features.append(np.median(nm.get('section_lengths', neuron)))
    features.append(max(nm.get('section_lengths', neuron)))
    features.append(min(nm.get('section_lengths', neuron)))
    features.append(max(nm.get('neurite_lengths', neuron)))
    features.append(nm.get('n_bifurcation_points', neuron))
    features.append(nm.get('n_segments', neuron))
    features.append(max(nm.get('section_tortuosity', neuron)))
    features.append(np.median(nm.get('section_tortuosity', neuron)))
    features.append(max(nm.get('remote_bifurcation_angles', neuron)))
    features.append(np.mean(nm.get('remote_bifurcation_angles', neuron)))
    features.append(min(nm.get('remote_bifurcation_angles', neuron)))
    features = np.array(features)
    features = features.reshape(((1,)+features.shape))
    return features

def GenerateH5py(dir_list):
    label7 = {'amacrine': 0, 'aspiny': 1, 'basket': 2, 'bipolar': 3, 'pyramidal': 4, 'spiny': 5, 'stellate': 6}
    i = 0
    for filename in os.listdir(dir_list):
        if filename.split('.')[-1] != 'swc': continue
        print(dir_list.split('/')[-1], '/', filename, ' ', i)
        if i == 0:
            datas = ReturnFeatures(nm.load_neuron(dir_list+'/'+filename))
            i = i + 1
            continue
        try:data = ReturnFeatures(nm.load_neuron(dir_list+'/'+filename))
        except:
            continue
        datas = np.concatenate((datas, data))
        i = i + 1
    datas = np.array(datas)
    labels = np.ones((datas.shape[0], 1)) * int(label7[dir_list.split('/')[-1]])
    return datas, labels

def GenerateMorphDataset(neuron_list,proportion):
    for i, neuron_type in enumerate(os.listdir(neuron_list)):
        if i == 0:
            datas, labels = GenerateH5py(neuron_list + '/' + neuron_type)
            continue
        data, label = GenerateH5py(neuron_list + '/' + neuron_type)
        datas = np.concatenate((datas, data))
        labels = np.concatenate((labels, label))
    print(datas.shape, ' ', labels.shape)
    state = np.random.get_state()
    np.random.shuffle(datas)
    np.random.set_state(state)
    np.random.shuffle(labels)
    datas = datas.astype(np.float32)
    labels = labels.astype(np.uint8)
    WriteH5py(dir=r'./MorphTrainDatasets.h5', data=datas[0:math.ceil(proportion * datas.shape[0])],
              label=labels[0:math.ceil(proportion * labels.shape[0])])
    WriteH5py(dir=r'./MorphTestDatasets.h5', data=datas[math.ceil(proportion * datas.shape[0]):-1],
              label=labels[math.ceil(proportion * labels.shape[0]):-1])

class Mlp(nn.Module):
    def __init__(self, num_classes=7):
        super(Mlp, self).__init__()
        self.linear1 = nn.Linear(16, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(self.bn1(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp3(x)
        x = self.linear3(x)
        return x


def train():
    train_dataset = DataSet(train=True,train_dir='./MorphTrainDatasets.h5',norm=False)
    test_dataset = DataSet(train=False,test_dir='./MorphTestDatasets.h5',norm=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2,
                                              drop_last=True)
    model = Mlp().to('cuda')
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(opt, step_size=20, gamma=0.5)
    criterion_CrossEntropy = nn.CrossEntropyLoss()
    best_test_acc = 0
    best_test_avg_acc = 0
    for epoch in range(50):
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        tqdm_batch = tqdm.tqdm(train_loader, desc='Epoch-{} training'.format(epoch))
        for data, label in tqdm_batch:
            data = data.type(torch.FloatTensor)
            label = label.type(torch.LongTensor)
            data, label = data.to('cuda'), label.to('cuda').squeeze()
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            preds = logits.max(dim=1)[1]
            count += batch_size
            loss = criterion_CrossEntropy(logits, label)
            loss.backward()
            opt.step()
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        tqdm_batch.close()

        if opt.param_groups[0]['lr'] > 1e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 1e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        print('Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch, train_loss * 1.0 / count,
                                                                              metrics.accuracy_score(train_true, train_pred),
                                                                              metrics.balanced_accuracy_score(train_true, train_pred)))

        with torch.no_grad():
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            tqdm_batch = tqdm.tqdm(test_loader, desc='Epoch-{} testing'.format(epoch))
            for data, label in tqdm_batch:
                data = data.type(torch.FloatTensor)
                label = label.type(torch.LongTensor)
                data, label = data.to('cuda'), label.to('cuda').squeeze()
                batch_size = data.size()[0]
                logits = model(data)
                preds = logits.max(dim=1)[1]
                loss = criterion_CrossEntropy(logits, label)
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            tqdm_batch.close()
            print('Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,test_loss * 1.0 / count,test_acc,avg_per_class_acc))
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_test_avg_acc = avg_per_class_acc
                torch.save(model.state_dict(), './Mlp(512-256).t7')
        print('best acc: ', best_test_acc, ' best avg acc: ', best_test_avg_acc)

def GenerateMorphDatabase(neuron_list):
    NeuronVectorDatabase = {}
    for neuron_type in os.listdir(neuron_list):
        for swc in os.listdir(neuron_list+'/'+neuron_type):
            print(swc)
            try:
                feature = ReturnFeatures(nm.load_neuron(neuron_list+'/'+neuron_type+'/'+swc))
                feature = feature[0,:]
                print(feature.shape)
            except: continue
            NeuronVectorDatabase[neuron_type + '/' + swc] = feature
    print(len(NeuronVectorDatabase))
    np.save('Morphometrics.npy', NeuronVectorDatabase)

def Replace(dir1,dir2):
    flag = 0
    with open(dir1,'r') as f,open(dir2,'w') as f2:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if line[0] == '#'or line[0] == '\n':
                continue
            if line[-3] == '-' and flag != 0:
                line = line[:-3] + line[-2:]
            f2.write(line)
            flag = flag+1
        f.close()
        f2.close()
    os.remove(dir1)
    os.rename(dir2,dir1)

def GenerateMorphometricsAddMorphoGNNDatabase(mmDatabase,morphognnDatabase,proportion=0.7):
    '''train a mlp through morphometrics added with the features of MorphoGNN to classify'''
    mmdatabase = np.load(mmDatabase, allow_pickle=True).item()
    morphognndatabase = np.load(morphognnDatabase, allow_pickle=True).item()
    label7 = {'amacrine': 0, 'aspiny': 1, 'basket': 2, 'bipolar': 3, 'pyramidal': 4, 'spiny': 5, 'stellate': 6}
    datas = []
    labels = []
    for neuron in mmdatabase.keys():
        print(neuron)
        datas.append(np.concatenate((mmdatabase[neuron], morphognndatabase[neuron])))
        type = neuron[0:neuron.find('/')]
        labels.append(label7[type])

    datas = np.array(datas)
    labels = np.array(labels)
    print(datas.shape)
    print(labels.shape)
    state = np.random.get_state()
    np.random.shuffle(datas)
    np.random.set_state(state)
    np.random.shuffle(labels)
    datas = datas.astype(np.float32)
    labels = labels.astype(np.uint8)
    WriteH5py(dir=r'./Morph+MorphoGNNTrainDatasets.h5', data=datas[0:math.ceil(proportion * datas.shape[0])],
              label=labels[0:math.ceil(proportion * labels.shape[0])])
    WriteH5py(dir=r'./Morph+MorphoGNNTestDatasets.h5', data=datas[math.ceil(proportion * datas.shape[0]):-1],
              label=labels[math.ceil(proportion * labels.shape[0]):-1])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate morphometric dataset')
    parser.add_argument('--swc_dir', type=str, default='./neuron7', help='file path of .swc files')
    args = parser.parse_args()
    GenerateMorphDataset(neuron_list=args.swc_dir,proportion=0.7)
    train()