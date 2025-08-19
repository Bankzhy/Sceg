import os
import dgl
import json
import torch
from pathlib import Path
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info

class FEDataset(DGLDataset):
    def __init__(self,
                 url=None,
                 split="train",
                 raw_dir=None,
                 force_reload=False,
                 verbose=False):
        self.graphs = []
        self.neg_graphs = []
        self.labels = []
        self.split = split
        super(FEDataset, self).__init__(name='fe_dataset',
                                         url=url,
                                         raw_dir=raw_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def download(self):
        # 将原始数据下载到本地磁盘
        pass

    def process(self):
        # 将原始数据处理为图、标签和数据集划分的掩码
        self.graphs, self.neg_graphs, self.labels = self._load_graph()

    def _load_graph(self):
        graphs = []
        neg_graphs = []
        labels = []

        pos_path = Path("dataset/fe/"+self.split + "_1.txt")
        neg_path = Path("dataset/fe/"+self.split + "_0.txt")
        failed_num = 0

        with open(pos_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                graph = json.loads(line)
                new_graph = self._get_graph(graph, 1)
                new_neg_graph = self._get_graph(graph, 0)
                graphs.append(new_graph)
                neg_graphs.append(new_neg_graph)
                labels.append(1)

        with open(neg_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                graph = json.loads(line)
                new_graph = self._get_graph(graph, 0)
                new_neg_graph = self._get_graph(graph, 1)
                graphs.append(new_graph)
                neg_graphs.append(new_neg_graph)
                labels.append(0)

        labels = torch.tensor(labels)
        self.num_classes = 2
        print("Failed:", failed_num)
        return graphs, neg_graphs, labels