import dgl
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F


class GATRGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feats, hid_feats, num_heads=2)
            for rel in rel_names}, aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hid_feats*2, out_feats, 1)
            for rel in rel_names}, aggregate='mean')

    def forward(self, graph, inputs):
        # inputs是节点的特征
        h = self.conv1(graph, inputs)

        # h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.relu(v.flatten(1)) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: v.squeeze(1) for k, v in h.items()}
        return h

class GATHeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):

        super().__init__()
        self.gat = GATRGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):

        h = g.ndata['feat']
        h = self.gat(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # 通过平均读出值来计算单图的表征
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)

                # if ntype == "method":
                #     hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                # else:
                #     hg = hg + dgl.max_nodes(g, 'h', ntype=ntype) * h["statement"].size(0)
            return self.classify(hg)
