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
from dataset.lm.lmd_dataset import LMDDataset
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import classification_report


from dataset.lc.lcd_dataset import LCDDataset
from dataset.lc.lcr_dataset import LCRDataset
from models.gat import GATHeteroClassifier, GATRGCN
from models.gcn import GCNHeteroClassifier, RGCN
from models.sage import SageHeteroClassifier, SageRGCN

dataset_path = Path(r"dataset/lc")
db = pymysql.connect(
    host="47.113.220.80",
    user="root",
    password="Apple3328823%",
    database="sce",
    charset="utf8mb4",  # Use utf8mb4 for full Unicode support
    connect_timeout=50
)

def collate(samples):
    '''
    将多个小图生成批次
    :param samples: 由图和标签的list组成（ graphs, labels）
    :return: batch_graph 也是一个图。
    这意味着任何适用于一个图的代码都可立即用于一批图。
    更重要的是，由于DGL并行处理所有节点和边缘上的消息，因此大大提高了效率。
    '''
    # print(samples)
    # graphs, labels = map(list, zip(*samples))
    # batch_graph = dgl.batch(graphs)
    # return batch_graph, torch.tensor(labels)
    graphs, labels = zip(*samples)
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels

def collate_r(samples):
    graphs, labels = zip(*samples)
    return graphs, labels

def build_training_dataset():
    if os.path.exists(dataset_path / "train_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "train_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading training 1...")
        cursor.execute("SELECT * FROM lc_master where `label`=1 and split='train'")
        for row in cursor.fetchall():
            lc_graph = row[7]
            f.write(lc_graph + "\n")
            pos_count += 1

    with open(dataset_path / "train_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM lc_master where `label`=0 and split='train' limit " + str(pos_count) )
        for row in cursor.fetchall():
            lc_graph = row[7]
            f.write(lc_graph + "\n")
        f.close()

def build_eval_dataset():

    if os.path.exists(dataset_path / "test_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "test_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading test 1...")
        cursor.execute("SELECT * FROM lc_master where `label`=1 and split='eval'")
        for row in cursor.fetchall():
            lc_graph = row[7]
            f.write(lc_graph + "\n")
            pos_count += 1

    with open(dataset_path / "test_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM lc_master where `label`=0 and split='eval' limit " + str(pos_count) )
        for row in cursor.fetchall():
            lc_graph = row[7]
            f.write(lc_graph + "\n")
        f.close()

def lc_detect():
    model_output = "output/model/lcd-model-gcn.pkl"
    input_dim = 12
    hidden_dim = 64
    set_epoch = 20

    build_training_dataset()
    build_eval_dataset()
    lcs_train = LCDDataset(split='train', raw_dir="output")
    lcs_test = LCDDataset(split='test', raw_dir="output")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data_loader = DataLoader(
        lcs_train,
        batch_size=32,
        shuffle=True,
        collate_fn=collate,
        drop_last=False)

    test_data_loader = GraphDataLoader(
        lcs_test,
        shuffle=True,
        batch_size=32,
        drop_last=False)

    g, l = lcs_train[0]
    etypes = g.etypes

    # model = GCNHeteroClassifier(input_dim, hidden_dim, 2, etypes)
    # model = SageHeteroClassifier(input_dim, hidden_dim, 2, etypes)
    model = GATHeteroClassifier(input_dim, hidden_dim, 2, etypes)


    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    model.train()
    epoch_losses = []
    for epoch in range(set_epoch):
        epoch_loss = 0
        for iter, (batched_graph, labels) in enumerate(train_data_loader):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            prediction = model(batched_graph)
            loss = loss_func(prediction, labels.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

    # 保存
    torch.save(model, model_output)
    # 加载
    model = torch.load(model_output)
    model.to("cpu")

    model.eval()
    y_true = []
    y_pred = []
    for batched_graph, labels in test_data_loader:
        pred = model(batched_graph)
        y_pred.extend(pred.argmax(1).tolist())
        y_true.extend(labels.tolist())
    target_names = ["neg", "pos"]
    print(classification_report(y_true, y_pred, target_names=target_names))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC:", auc)


def lc_refact():
    model_output = "output/model/lcr-model-gat.pkl"
    hidden_dim = 64
    set_epoch = 20

    build_training_dataset()
    build_eval_dataset()

    lms_train = LCRDataset(split='train', raw_dir="output")
    lms_test = LCRDataset(split='test', raw_dir="output")

    train_data_loader = GraphDataLoader(
        lms_train,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_r,
        drop_last=False)

    test_data_loader = GraphDataLoader(
        lms_test,
        shuffle=True,
        batch_size=32,
        drop_last=False)

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
        y_pred.extend(prediction.argmax(1).tolist())
        y_true.extend(st_labels.cpu())

    target_names = ["neg", "pos"]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    # save_report(report, report_save_path)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC:", auc)

def test_detect():
    model_output = "output/model/lcd-model-gcn.pkl"
    lcs_test = LCDDataset(split='test', raw_dir="output")

    test_data_loader = GraphDataLoader(
        lcs_test,
        shuffle=True,
        batch_size=32,
        drop_last=False)

    model = torch.load(model_output)
    model.to("cpu")

    model.eval()
    y_true = []
    y_pred = []
    for batched_graph, labels in test_data_loader:
        pred = model(batched_graph)
        y_pred.extend(pred.argmax(1).tolist())
        y_true.extend(labels.tolist())
    target_names = ["neg", "pos"]
    print(classification_report(y_true, y_pred, target_names=target_names))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC:", auc)

def test_refact():

    model_output = "output/model/lcr-model-gcn.pkl"

    lms_test = LCRDataset(split='test', raw_dir="output")

    test_data_loader = GraphDataLoader(
        lms_test,
        shuffle=True,
        batch_size=32,
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
        y_pred.extend(prediction.argmax(1).tolist())
        y_true.extend(st_labels.cpu())

    target_names = ["neg", "pos"]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    # save_report(report, report_save_path)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC:", auc)


if __name__ == '__main__':
    # lc_detect()
    # test_detect()
    lc_refact()
    # test_refact()