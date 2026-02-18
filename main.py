import numpy as np
import torch
from utils.load_data import *
from utils.params import get_params
from utils.Config import Config
from module.SAGA import *
from module.preprocess import *
from utils.load_data import random_split
import warnings
import datetime
import time
import random
from torch.utils.data import RandomSampler
import matplotlib.pyplot as plt
from sklearn import metrics
import os

warnings.filterwarnings('ignore')
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(config, args, device):
    feat_s, adjs_s, label_s, feat_t, adjs_t, label_t = load_data(args.dataset)
    nb_classes = label_s.shape[-1]

    #划分训练集和测试集，没有用到验证集
    train_mask_s, val_mask_s, test_mask_s = random_split(feat_s, label_s, adjs_s)
    train_mask_t, val_mask_t, test_mask_t = random_split(feat_t, label_t, adjs_t)
    train_label_s = label_s[train_mask_s]

    print("nb_classes:",nb_classes)
    num_source_node = len(feat_s[train_mask_s])
    num_target_node = len(feat_t[train_mask_t])

    feats_dim = feat_s.shape[1]
    sub_num_s = int(len(adjs_s))
    sub_num_t = int(len(adjs_t))
    print("The number of source meta-paths: ", sub_num_s)
    print("The number of target meta-paths: ", sub_num_t)
    print("Number of training source nodes:", num_source_node)
    print("Number of training target nodes:", num_target_node)
    label_s = label_s.to(device)
    label_t = label_t.to(device)
    if torch.cuda.is_available():
        print('Using CUDA')
        adjs_s = [adj.cuda() for adj in adjs_s]
        feat_s = feat_s.cuda()
        adjs_t = [adj.cuda() for adj in adjs_t]
        feat_t = feat_t.cuda()

    train_label_s = label_s[train_mask_s]
    train_feat_s = feat_s[train_mask_s]
    if(is_sparse_coo(adjs_s[0])):
        train_adjs_s = [slice_sparse_matrix_with_mask(train_mask_s, adj_s, device) for adj_s in adjs_s]
    else:
        train_indices = torch.where(train_mask_s == 1)[0]
        train_adjs_s = [adj_s[train_indices][:, train_indices] for adj_s in adjs_s]

    train_label_t = label_t[train_mask_t]
    train_feat_t = feat_t[train_mask_t]
    if (is_sparse_coo(adjs_t[0])):
        train_adjs_t = [slice_sparse_matrix_with_mask(train_mask_t, adj_t, device) for adj_t in adjs_t]
    else:
        train_indices = torch.where(train_mask_t == 1)[0]
        train_adjs_t = [adj_t[train_indices][:, train_indices].to(torch.float32) for adj_t in adjs_t]

    train_adjs_o_s = graph_process_large(train_adjs_s, train_feat_s, args)
    train_adjs_o_t = graph_process_large(train_adjs_t, train_feat_t, args)

    train_f_list_s = APPNP([train_feat_s for _ in range(sub_num_s)], train_adjs_o_s, config.hyperparameters['k_hop'], args.filter_alpha)
    train_f_list_t = APPNP([train_feat_t for _ in range(sub_num_t)], train_adjs_o_t, config.hyperparameters['k_hop'], args.filter_alpha)

    dominant_index_s = pre_compute_dominant_view_large(train_f_list_s, train_feat_s)
    dominant_index_t = pre_compute_dominant_view_large(train_f_list_t, train_feat_t)

    print("Started training...")
    #target_acc_list = []

    seed = args.seed
    set_seed(seed)
    model = SAGA(feats_dim, sub_num_s, args.hidden_dim, args.embed_dim, args.tau, args.dropout, len(feat_s), dominant_index_s, device, nb_classes, config)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        model.cuda()

    period = 2
    batchsize = args.batchsize
    batch_num_s = math.ceil(num_source_node / batchsize)
    batch_num_t = math.ceil(num_target_node / batchsize)
    print("batch_num_s: ", batch_num_s, "batch_num_t: ", batch_num_t)
    max_batch_num = max(batch_num_s, batch_num_t)
    starttime = datetime.datetime.now()

    source_accs = []
    target_accs = []

    if args.load_parameters == False:
        t0 = time.time()
        best_source_acc = 0.0
        best_target_acc = 0.0
        best_epoch = 0.0
        for epoch in range(args.nb_epochs):
            model.train()
            sampler_s = RandomSampler(range(num_source_node), replacement=False)
            sampler_s = torch.tensor([i for i in sampler_s])#.to(device)
            sampler_t = RandomSampler(range(num_target_node), replacement=False)
            sampler_t = torch.tensor([i for i in sampler_t])#.to(device)

            for batch_index in range(max_batch_num):
                # ---- Source 的索引范围 ----
                start_index_s = batch_index * batchsize
                end_index_s = start_index_s + batchsize
                if end_index_s > num_source_node:
                    end_index_s = num_source_node

                # 当 batch_index 超过 batch_num_s 时，可能意味着没有更多 source 节点可供采样
                if start_index_s >= num_source_node:
                    seed_node_s = None
                else:
                    seed_node_s = sampler_s[start_index_s:end_index_s]

                # ---- Target 的索引范围 ----
                start_index_t = batch_index * batchsize
                end_index_t = start_index_t + batchsize
                if end_index_t > num_target_node:
                    end_index_t = num_target_node

                # 当 batch_index 超过 batch_num_t 时，可能意味着没有更多 target 节点可供采样
                if start_index_t >= num_target_node:
                    seed_node_t = None
                else:
                    seed_node_t = sampler_t[start_index_t:end_index_t]

                # 如果 seed_node_s 或 seed_node_t 不为空，则取出对应特征
                if seed_node_s is not None and len(seed_node_s) > 0:
                    feat_sampler_s = train_feat_s[seed_node_s]
                    f_list_sampler_s = [f[seed_node_s] for f in train_f_list_s]
                    label_sampler_s = train_label_s[seed_node_s]
                else:
                    feat_sampler_s = None
                    f_list_sampler_s = None
                    label_sampler_s = None

                if seed_node_t is not None and len(seed_node_t) > 0:
                    feat_sampler_t = train_feat_t[seed_node_t]
                    f_list_sampler_t = [f[seed_node_t] for f in train_f_list_t]
                    label_sampler_t = train_label_t[seed_node_t]
                else:
                    feat_sampler_t = None
                    f_list_sampler_t = None
                    label_sampler_t = None

                optimizer.zero_grad()
                loss = model(feat_sampler_t, f_list_sampler_t, feat_sampler_s, f_list_sampler_s, label_sampler_s, False)
                loss.backward()
                optimizer.step()

            #这里加入evaluate
            test_mask_s = train_mask_s
            test_mask_t = train_mask_t

            test_label_s = label_s[test_mask_s]
            test_feat_s = feat_s[test_mask_s]
            test_adjs_s = [slice_sparse_matrix_with_mask(test_mask_s, adj_s, device) for adj_s in adjs_s]

            test_label_t = label_t[test_mask_t]
            test_feat_t = feat_t[test_mask_t]
            test_adjs_t = [slice_sparse_matrix_with_mask(test_mask_t, adj_t, device) for adj_t in adjs_t]

            test_adjs_o_s = graph_process_large(test_adjs_s, test_feat_s, args)
            test_adjs_o_t = graph_process_large(test_adjs_t, test_feat_t, args)

            test_f_list_s = APPNP([test_feat_s for _ in range(sub_num_s)], test_adjs_o_s, config.hyperparameters['k_hop'],
                                   args.filter_alpha)
            test_f_list_t = APPNP([test_feat_t for _ in range(sub_num_t)], test_adjs_o_t, config.hyperparameters['k_hop'],
                                   args.filter_alpha)
            model.eval()
            if test_f_list_s is not None:
                test_z_list_s = model.encoder(test_f_list_s)
            embedding_s = torch.cat(test_z_list_s, dim=1)  # [batch_size, embed_dim * sub_num]
            logits_s = model.classifier(embedding_s)  # [batch_size, nb_classes]
            preds_s = logits_s.argmax(dim=1)
            corrects_s = preds_s.eq(test_label_s.argmax(dim=1))
            accuracy_s = corrects_s.float().mean().cpu()

            if test_f_list_t is not None:
                test_z_list_t = model.encoder(test_f_list_t)
            embedding_t = torch.cat(test_z_list_t, dim=1)  # [batch_size, embed_dim * sub_num]
            logits_t = model.classifier(embedding_t)  # [batch_size, nb_classes]
            preds_t = logits_t.argmax(dim=1)
            corrects_t = preds_t.eq(test_label_t.argmax(dim=1))
            accuracy_t = corrects_t.float().mean().cpu()
            source_accs.append(accuracy_s)
            target_accs.append(accuracy_t)

            true_label_s = torch.argmax(test_label_s, dim=1)
            true_label_t = torch.argmax(test_label_t, dim=1)

            # 计算 F1_macro 分数
            f1_macro_s = metrics.f1_score(true_label_s.cpu().numpy(), preds_s.cpu().numpy(), average='macro')
            f1_macro_t = metrics.f1_score(true_label_t.cpu().numpy(), preds_t.cpu().numpy(), average='macro')

            print("Epoch: {}, source_acc: {:.4f}, target_acc: {:.4f}, source_macroF1: {:.4f}, target_macroF1: {:.4f} ".format(epoch, accuracy_s, accuracy_t, f1_macro_s, f1_macro_t))
            if accuracy_t > best_target_acc:
                best_target_acc = accuracy_t
                best_source_acc = accuracy_s
                best_epoch = epoch

        t1 = time.time()
        training_time = t1 - t0
        training_time = format_time(training_time)
        print("Training Time:", training_time)
        checkpoint_dir = f'./checkpoint/{args.dataset}'
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'best_{seed}.pth')
        torch.save(model.state_dict(), checkpoint_path)

    else:
        model.load_state_dict(torch.load('./best/' + args.dataset + '/best_' + str(0) + '.pth'))

    print("=============================================================")
    line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}" \
        .format(id, best_epoch, best_source_acc, best_target_acc)

    print(line)

def main():
    # Get parameters directly from command line
    args = get_params()
    config_file = "./config/" + str(args.dataset) + ".ini"
    config = Config(config_file)
    # Setup device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.gpu < num_gpus:
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(args.gpu)
            print(f"\nUsing GPU: {args.gpu} - {torch.cuda.get_device_name(args.gpu)}")
        else:
            print(f"Warning: GPU {args.gpu} does not exist, using GPU 0")
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Print all parameters
    print("\nTraining parameters:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Train
    train(config, args, device)



if __name__ == "__main__":
    main()
