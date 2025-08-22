import os
from pathlib import Path

import dgl
import pymysql
import torch
from pathlib import Path
import pymysql
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader

from dataset.fe.fe_dataset import FEDataset
from dataset.lm.lmd_dataset import LMDDataset
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import classification_report


from dataset.lc.lcd_dataset import LCDDataset
from dataset.lc.lcr_dataset import LCRDataset
from models.gat import GATHeteroClassifier, GATRGCN
from models.gcn import GCNHeteroClassifier, RGCN
from models.sage import SageHeteroClassifier, SageRGCN


def collate_r(samples):
    graphs, labels = zip(*samples)
    return graphs, labels

def fe_refact():
    model_output = "output/model/fed-model-gat.pkl"
    hidden_dim = 64
    set_epoch = 40

    lms_train = FEDataset(split='train', raw_dir="output")
    # lms_test = FEDataset(split='test', raw_dir="output")

    train_data_loader = GraphDataLoader(
        lms_train,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_r,
        drop_last=False)

    # test_data_loader = GraphDataLoader(
    #     lms_test,
    #     shuffle=True,
    #     batch_size=32,
    #     drop_last=False)

    g, l = lms_train[0]
    etypes = g.etypes
    n_st_classes = 2
    st_feats = g.nodes['method'].data['feat']
    n_hetero_features = len(st_feats[0])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # model = RGCN(n_hetero_features, hidden_dim, n_st_classes, etypes)
    # model = SageRGCN(n_hetero_features, hidden_dim, n_st_classes, etypes)
    model = GATRGCN(n_hetero_features, hidden_dim, n_st_classes, etypes)


    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    model.train()
    epoch_losses = []

    for epoch in range(set_epoch):
        epoch_loss = 0
        for iter, (batched_graph, labels) in enumerate(train_data_loader):
            for graph in batched_graph:
                st_feats = graph.nodes['method'].data['feat']
                st_labels = graph.nodes['method'].data['label']
                node_features = {'method': st_feats.to(device)}
                graph = graph.to(device)
                st_labels = st_labels.to(device)
                model.train()
                prediction = model(graph, node_features)['method']
                loss = loss_func(prediction, st_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

    # 保存
    torch.save(model, model_output)

    eval_fe(model_output)

    # # 加载
    # model = torch.load(model_output)
    # model.to("cpu")
    #
    #
    # model.eval()
    # y_true = []
    # y_pred = []
    # for graph, labels in test_data_loader:
    #     st_feats = graph.nodes['method'].data['feat']
    #     st_labels = graph.nodes['method'].data['label']
    #     node_features = {'method': st_feats}
    #     prediction = model(graph, node_features)['method']
    #     y_pred.extend(prediction.argmax(1).tolist())
    #     y_true.extend(st_labels.cpu())
    #
    # target_names = ["neg", "pos"]
    # report = classification_report(y_true, y_pred, target_names=target_names)
    # print(report)
    # # save_report(report, report_save_path)
    # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # print("AUC:", auc)

def eval_fe(model_output):
    # model_output = "output/model/fed-model-gcn.pkl"
    lms_test = FEDataset(split='test', raw_dir="output")

    test_data_loader = GraphDataLoader(
        lms_test,
        shuffle=False,
        batch_size=1,
        drop_last=False)

    # 加载
    model = torch.load(model_output)
    model.to("cpu")


    model.eval()

    y_true = []
    y_pred = []
    for graph, labels in test_data_loader:
        st_feats = graph.nodes['method'].data['feat']
        st_labels = graph.nodes['method'].data['label']
        node_features = {'method': st_feats}
        prediction = model(graph, node_features)['method']
        cls_pred = prediction.argmax(1).tolist()
        cls_label = st_labels.cpu()

        y_pred.append(cls_pred)
        y_true.append(cls_label)

    target_names = ["neg", "pos"]
    all_fe = []
    with open("dataset/fe/fe_all.txt", "r", encoding="utf-8") as f:
        for l in f.readlines():
            el = l.replace("\n","")
            all_fe.append(el)


    new_pred = []
    new_label = []
    for ci, cls in enumerate(lms_test.class_list):
        for mi, method in enumerate(lms_test.method_list[ci]):
            pred = y_pred[ci][mi]
            label = y_true[ci][mi]
            content = cls+","+method
            if content in all_fe:
                new_pred.append(pred)
                new_label.append(torch.tensor(label))


    report = classification_report(new_label, new_pred, target_names=target_names)
    print(report)


if __name__ == '__main__':
    fe_refact()