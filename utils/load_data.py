import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from sklearn.preprocessing import OneHotEncoder
import scipy.io as sio
from module.preprocess import remove_self_loop, find_idx
import pickle as pkl
import torch.nn.functional as F

def random_split(feat, labels, adjs):
    """
    给数据集随机生成train set、val set、test set的mask，用于划分训练集、验证集、测试集.目前1:0:0
    source数据和target数据都用到
    """
    # 打乱索引
    num_samples = feat.size(0)
    random_indices = torch.randperm(num_samples)

    # 定义划分比例
    train_ratio = 1
    val_ratio = 0
    test_ratio = 0

    # 计算每个数据集的样本数量
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)

    # 获取训练、验证和测试的索引
    train_indices = random_indices[:train_size]
    val_indices = random_indices[train_size:train_size + val_size]
    test_indices = random_indices[train_size + val_size:]

    # 创建掩码
    train_mask = torch.zeros(num_samples, dtype=torch.uint8)
    val_mask = torch.zeros(num_samples, dtype=torch.uint8)
    test_mask = torch.zeros(num_samples, dtype=torch.uint8)

    train_mask[train_indices] = 1
    val_mask[val_indices] = 1
    test_mask[test_indices] = 1
    return train_mask, val_mask, test_mask

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return th.sparse.FloatTensor(index, th.ones(len(index[0])), adj.shape)

def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def sp_tensor_to_sp_csr(adj):
    adj = adj.coalesce()
    row = adj.indices()[0]
    col = adj.indices()[1]
    data = adj.values()
    shape = adj.size()
    adj = sp.csr_matrix((data, (row, col)), shape=shape)
    return adj

def convert_to_sparse(tensor):
    """
    dtype:float32
    """
    # 获取非零元素的索引和对应的值
    indices = tensor.nonzero().t()  # 转置以符合 COO 格式
    values = tensor[tensor != 0]

    # 创建稀疏 COO 张量
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=tensor.size())
    return sparse_tensor


def load_ACM1_ACM2():
    path_s = f"./data/ACM1/"
    label_s = torch.load(path_s+'label.pt')
    feat_s = torch.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'psp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/ACM2/"
    label_t = torch.load(path_t + 'label.pt')
    feat_t = torch.load(path_t + 'feat.pt').float()
    adj_1_t = th.load(path_t + 'pap.pt').coalesce()
    adj_2_t = th.load(path_t + 'psp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_ACM2_ACM1():
    path_s = f"./data/ACM2/"
    label_s = torch.load(path_s+'label.pt')
    feat_s = torch.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s + 'pap.pt').coalesce()
    adj_2_s = th.load(path_s + 'psp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/ACM1/"
    label_t = torch.load(path_t + 'label.pt')
    feat_t = torch.load(path_t + 'feat.pt').float()
    adj_1_t = th.load(path_t + 'pap.pt').coalesce()
    adj_2_t = th.load(path_t + 'psp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_CN_US():
    path_s = f"./data/CN/"
    label_s = th.load(path_s+'label.pt')
    label_s = F.one_hot(label_s, num_classes=10)

    feat_s = th.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'pp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/US/"
    label_t = th.load(path_t+'label.pt')
    label_t = F.one_hot(label_t, num_classes=10)

    feat_t = th.load(path_t+'feat.pt').float()
    adj_1_t = th.load(path_t+'pap.pt').coalesce()
    adj_2_t = th.load(path_t+'pp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_US_CN():
    path_s = f"./data/US/"
    label_s = th.load(path_s+'label.pt')
    label_s = F.one_hot(label_s, num_classes=10)

    feat_s = th.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'pp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/CN/"
    label_t = th.load(path_t+'label.pt')
    label_t = F.one_hot(label_t, num_classes=10)

    feat_t = th.load(path_t+'feat.pt').float()
    adj_1_t = th.load(path_t+'pap.pt').coalesce()
    adj_2_t = th.load(path_t+'pp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_JP_CN():
    path_s = f"./data/JP/"
    label_s = th.load(path_s+'label.pt')
    label_s = F.one_hot(label_s, num_classes=10)

    feat_s = th.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'pp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/CN/"
    label_t = th.load(path_t+'label.pt')
    label_t = F.one_hot(label_t, num_classes=10)

    feat_t = th.load(path_t+'feat.pt').float()
    adj_1_t = th.load(path_t+'pap.pt').coalesce()
    adj_2_t = th.load(path_t+'pp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_CN_JP():
    path_s = f"./data/CN/"
    label_s = th.load(path_s+'label.pt')
    label_s = F.one_hot(label_s, num_classes=10)

    feat_s = th.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'pp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/JP/"
    label_t = th.load(path_t+'label.pt')
    label_t = F.one_hot(label_t, num_classes=10)

    feat_t = th.load(path_t+'feat.pt').float()
    adj_1_t = th.load(path_t+'pap.pt').coalesce()
    adj_2_t = th.load(path_t+'pp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_DE_FR():
    path_s = f"./data/DE/"
    label_s = th.load(path_s+'label.pt')
    label_s = F.one_hot(label_s, num_classes=10)

    feat_s = th.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'pp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/FR/"
    label_t = th.load(path_t+'label.pt')
    label_t = F.one_hot(label_t, num_classes=10)

    feat_t = th.load(path_t+'feat.pt').float()
    adj_1_t = th.load(path_t+'pap.pt').coalesce()
    adj_2_t = th.load(path_t+'pp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_FR_DE():
    path_s = f"./data/FR/"
    label_s = th.load(path_s+'label.pt')
    label_s = F.one_hot(label_s, num_classes=10)

    feat_s = th.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'pp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/DE/"
    label_t = th.load(path_t+'label.pt')
    label_t = F.one_hot(label_t, num_classes=10)

    feat_t = th.load(path_t+'feat.pt').float()
    adj_1_t = th.load(path_t+'pap.pt').coalesce()
    adj_2_t = th.load(path_t+'pp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_RU_US():
    path_s = f"./data/RU/"
    label_s = th.load(path_s+'label.pt')
    label_s = F.one_hot(label_s, num_classes=10)

    feat_s = th.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'pp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/US/"
    label_t = th.load(path_t+'label.pt')
    label_t = F.one_hot(label_t, num_classes=10)

    feat_t = th.load(path_t+'feat.pt').float()
    adj_1_t = th.load(path_t+'pap.pt').coalesce()
    adj_2_t = th.load(path_t+'pp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_US_RU():
    path_s = f"./data/US/"
    label_s = th.load(path_s+'label.pt')
    label_s = F.one_hot(label_s, num_classes=10)

    feat_s = th.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'pp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/RU/"
    label_t = th.load(path_t+'label.pt')
    label_t = F.one_hot(label_t, num_classes=10)

    feat_t = th.load(path_t+'feat.pt').float()
    adj_1_t = th.load(path_t+'pap.pt').coalesce()
    adj_2_t = th.load(path_t+'pp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t


def load_DE_CN():
    path_s = f"./data/DE/"
    label_s = th.load(path_s+'label.pt')
    label_s = F.one_hot(label_s, num_classes=10)

    feat_s = th.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'pp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/CN/"
    label_t = th.load(path_t+'label.pt')
    label_t = F.one_hot(label_t, num_classes=10)

    feat_t = th.load(path_t+'feat.pt').float()
    adj_1_t = th.load(path_t+'pap.pt').coalesce()
    adj_2_t = th.load(path_t+'pp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_CN_DE():
    path_s = f"./data/CN/"
    label_s = th.load(path_s+'label.pt')
    label_s = F.one_hot(label_s, num_classes=10)

    feat_s = th.load(path_s+'feat.pt').float()
    adj_1_s = th.load(path_s+'pap.pt').coalesce()
    adj_2_s = th.load(path_s+'pp.pt').coalesce()
    adjs_s = [adj_1_s, adj_2_s]

    path_t = f"./data/DE/"
    label_t = th.load(path_t+'label.pt')
    label_t = F.one_hot(label_t, num_classes=10)

    feat_t = th.load(path_t+'feat.pt').float()
    adj_1_t = th.load(path_t+'pap.pt').coalesce()
    adj_2_t = th.load(path_t+'pp.pt').coalesce()
    adjs_t = [adj_1_t, adj_2_t]

    return feat_s, adjs_s, label_s, feat_t, adjs_t, label_t

def load_data(dataset):
    if dataset == "ACM1-ACM2":
        data = load_ACM1_ACM2()
    elif dataset == "ACM2-ACM1":
        data = load_ACM2_ACM1()
    elif dataset == 'CN-US':
        data = load_CN_US()
    elif dataset == 'US-CN':
        data =  load_US_CN()
    elif dataset == 'JP-CN':
        data = load_JP_CN()
    elif dataset == 'CN-JP':
        data = load_CN_JP()
    elif dataset == 'DE-FR':
        data = load_DE_FR()
    elif dataset == 'FR-DE':
        data = load_FR_DE()
    elif dataset == 'RU-US':
        data = load_RU_US()
    elif dataset == 'US-RU':
        data = load_US_RU()
    elif dataset == 'DE-CN':
        data = load_DE_CN()
    elif dataset == 'CN-DE':
        data = load_CN_DE()
    else:
        print("unknown dataset")

    return data



