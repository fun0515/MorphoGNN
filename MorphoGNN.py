import torch
import torch.nn as nn
import torch.nn.init as init
import tqdm
import numpy as np
from dataset import DataSet
import torch.optim as optim
import sklearn.metrics as metrics
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class MorphoGNN(nn.Module):
    def __init__(self, num_classes=7):
        super(MorphoGNN, self).__init__()

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(32 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(96 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(224 * 2, 256 , kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(480, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = get_graph_feature(x, k=8)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x2 = get_graph_feature(x1, k=16)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1,x2),1)

        x3 = get_graph_feature(x, k=32)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), 1)

        x4 = get_graph_feature(x, k=64)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), 1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,-1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,-1)
        x = torch.cat((x1, x2), 1)

        self.feature = x
        x = self.linear1(x)
        x = F.leaky_relu(self.bn6(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp3(x)
        x = self.linear3(x)

        return self.feature,x


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.pow(x, 2).sum(dim=1)).view(bs1, 1).repeat(1, bs2)) * \
                (torch.sqrt(torch.pow(y, 2).sum(dim=1).view(1, bs2).repeat(bs1, 1)))
    cosine = frac_up / frac_down
    cos_d = 1 - cosine
    return cos_d


def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-100000.0) * (1 - mat_similarity), dim=1,
                                                       descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + 100000.0 * mat_similarity, dim=1,
                                                       descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5, normalize_feature=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, emb, label):
        if self.normalize_feature:
            emb = F.normalize(emb)

        mat_dist = euclidean_dist(emb, emb)

        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

        dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
        assert dist_an.size(0) == dist_ap.size(0)
        y = torch.ones_like(dist_ap)
        loss = self.margin_loss(dist_an, dist_ap, y)

        prec = (dist_an.data > dist_ap.data).sum() * 1.0 / y.size(0)
        return loss, prec

if __name__ == '__main__':
    train_dataset = DataSet(train=True)
    test_dataset = DataSet(train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2,
                                              drop_last=True)
    model = MorphoGNN().to('cuda')
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(opt, step_size=20, gamma=0.5)
    criterion_CrossEntropy = nn.CrossEntropyLoss()
    criterion_triple = TripletLoss()

    best_test_acc = 0
    best_test_avg_acc = 0
    for epoch in range(50):
        train_loss = 0.0
        train_triplet_loss = 0.0
        train_ce_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        tqdm_batch = tqdm.tqdm(train_loader, desc='Epoch-{} training'.format(epoch))
        for data, label in tqdm_batch:
            data = data.type(torch.FloatTensor)
            label = label.type(torch.LongTensor)
            data, label = data.to('cuda'), label.to('cuda').squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            features, logits = model(data)
            preds = logits.max(dim=1)[1]
            count += batch_size
            triplet_loss,_ = criterion_triple(features,label)
            ce_loss = criterion_CrossEntropy(logits,label)
            loss = triplet_loss * 1 + ce_loss
            loss.backward()
            opt.step()
            train_triplet_loss += triplet_loss.item() * batch_size
            train_ce_loss += ce_loss.item() * batch_size
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
        print('Train %d, loss: %.6f, triplet_loss: %.6f, CE_loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,train_loss * 1.0 / count,
                                                                              train_triplet_loss * 1.0/count, train_ce_loss * 1.0/count,
                                                                              metrics.accuracy_score(train_true, train_pred),
                                                                              metrics.balanced_accuracy_score(train_true, train_pred)))

        with torch.no_grad():
            test_loss = 0.0
            test_triplet_loss = 0.0
            test_ce_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            tqdm_batch = tqdm.tqdm(test_loader, desc='Epoch-{} testing'.format(epoch))
            for data, label in tqdm_batch:
                data = data.type(torch.FloatTensor)
                label = label.type(torch.LongTensor)
                data, label = data.to('cuda'), label.to('cuda').squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                features, logits = model(data)
                preds = logits.max(dim=1)[1]
                triplet_loss, _ = criterion_triple(features, label)
                ce_loss = criterion_CrossEntropy(logits, label)
                loss = triplet_loss * 1 + ce_loss
                count += batch_size
                test_loss += loss.item() * batch_size
                test_triplet_loss += triplet_loss.item() * batch_size
                test_ce_loss += ce_loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            tqdm_batch.close()
            print('Test %d, loss: %.6f, triplet_loss: %.6f, CE_loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                  test_loss*1.0/count,test_triplet_loss*1.0/count,
                                                                                  test_ce_loss*1.0/count,test_acc,
                                                                                  avg_per_class_acc))
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_test_avg_acc = avg_per_class_acc
                torch.save(model.state_dict(), './MorphoGNN.t7')
        print('best acc: ',best_test_acc,' best avg acc: ',best_test_avg_acc)
