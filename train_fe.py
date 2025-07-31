import os
import pymysql
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import random
import dgl.nn as dglnn
import dgl.function as fn
from pathlib import Path
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from dataset.fe.fe_dataset import FEDataset
from models.gcn import RGCN
from sklearn import metrics
from sklearn.metrics import classification_report


dataset_path = Path(r"dataset/fe")
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
    graphs, neg_graphs, labels = zip(*samples)
    batched_graph = dgl.batch(graphs)
    batched_neg_graph = dgl.batch(neg_graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_neg_graph, batched_labels
class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h, etype):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        # section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']

def build_training_dataset():

    if os.path.exists(dataset_path / "train_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "train_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading training 1...")
        cursor.execute("SELECT * FROM fe_master where `label`=1 and split='train'")
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
            pos_count += 1

    with open(dataset_path / "train_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM fe_master where `label`=0 and split='train' limit " + str(pos_count) )
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
        f.close()

def build_eval_dataset():

    if os.path.exists(dataset_path / "test_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "test_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading test 1...")
        cursor.execute("SELECT * FROM fe_master where `label`=1 and split='eval'")
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
            pos_count += 1

    with open(dataset_path / "test_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM fe_master where `label`=0 and split='eval' limit " + str(pos_count) )
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
        f.close()

def construct_negative_graph(graph, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    print(src)

def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def run():
    model_output = "output/model/fe-model-gcn.pkl"
    input_dim = 14
    hidden_dim = 64
    set_epoch = 8

    build_training_dataset()
    build_eval_dataset()
    lms_train = FEDataset(split='train', raw_dir="output")
    lms_test = FEDataset(split='test', raw_dir="output")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data_loader = DataLoader(
        lms_train,
        batch_size=32,
        shuffle=True,
        collate_fn=collate,
        drop_last=False)

    test_data_loader = GraphDataLoader(
        lms_test,
        shuffle=True,
        batch_size=32,
        drop_last=False)

    g, ng, l = lms_train[0]
    etypes = g.etypes


    model = Model(input_dim, hidden_dim, 5, etypes)
    model.to(device)
    epoch_losses = []
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(set_epoch):
        epoch_loss = 0
        for iter, (batched_graph, batched_neg_graph, labels) in enumerate(train_data_loader):
            batched_graph = batched_graph.to(device)
            batched_neg_graph = batched_neg_graph.to(device)
            labels = labels.to(device)
            class_feats = batched_graph.nodes['class'].data['feat']
            method_feats = batched_neg_graph.nodes['method'].data['feat']

            node_features = {'class': class_feats.to(device), 'method': method_feats.to(device)}
            pos_score, neg_score = model(batched_graph, batched_neg_graph, node_features, ('class', 'include', 'method'))
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
            # print(loss.item())
        epoch_loss /= (iter + 1)

        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

    # 保存
    torch.save(model, model_output)
    # 加载
    model = torch.load(model_output)

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

if __name__ == '__main__':
    run()