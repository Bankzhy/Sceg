import os
import dgl
import json
import torch
from pathlib import Path
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info

class FEDatasetOld(DGLDataset):
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
        super(FEDatasetOld, self).__init__(name='fe_dataset',
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

    def _get_graph(self, fe_graph, label):
        nodes = fe_graph["nodes"]

        method_nodes_ids = []
        class_nodes_ids = []

        class_feature = []
        nodes_feature = []
        include_edge_index = [[], []]

        for index, node in enumerate(nodes):
            if node["type"] == "class":
                feature = []
                feature.append(node["metrics"]["class_loc"])
                feature.append(node["metrics"]["nom"])
                feature.append(node["metrics"]["noa"])
                feature.append(node["metrics"]["cis"])
                feature.append(node["metrics"]["nopa"])
                feature.append(node["metrics"]["atfd"])
                feature.append(node["metrics"]["wmc"])
                feature.append(node["metrics"]["tcc"])
                feature.append(node["metrics"]["dcc"])
                feature.append(node["metrics"]["lcom"])
                feature.append(node["metrics"]["cam"])
                feature.append(node["metrics"]["dit"])
                feature.append(node["metrics"]["noam"])
                feature.append(node["metrics"]["dist"])
                class_feature.append(feature)
                class_nodes_ids.append(node["id"])
            elif node["type"] == "method":
                feature = []
                feature.append(node["metrics"]["loc"])
                feature.append(node["metrics"]["cc"])
                feature.append(node["metrics"]["pc"])
                feature.append(node["metrics"]["lcom1"])
                feature.append(node["metrics"]["lcom2"])
                feature.append(node["metrics"]["lcom3"])
                feature.append(node["metrics"]["tsmc"])
                feature.append(node["metrics"]["nbd"])
                feature.append(node["metrics"]["fuc"])
                feature.append(node["metrics"]["lmuc"])
                feature.append(node["metrics"]["nbd"])
                feature.append(node["metrics"]["noav"])
                feature.append(node["metrics"]["nfdi"])
                feature.append(node["metrics"]["nldi"])
                nodes_feature.append(feature)
                method_nodes_ids.append(node["id"])
        data_dict = {}

        if label == 1:
            data_dict[('class', 'include', 'method')] = (torch.tensor([1]), torch.tensor([0]))
        else:
            data_dict[('class', 'include', 'method')] = (torch.tensor([0]), torch.tensor([0]))

        num_nodes_dict = {'method': len(nodes_feature)}
        num_nodes_dict['class'] = 2
        data_dict[('class', 'include', 'class')] = (torch.tensor([0]), torch.tensor([0]))

        G = dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)

        G.nodes['class'].data['feat'] = torch.tensor(class_feature).to(torch.float32)
        G.nodes['method'].data['feat'] = torch.tensor(nodes_feature).to(torch.float32)
        print(G)

        return G

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 2

    def __getitem__(self, idx):
        # 通过idx得到与之对应的一个样本
        return self.graphs[idx], self.neg_graphs[idx], self.labels[idx]

    def __len__(self):
        # 数据样本的数量
        return len(self.graphs)

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, self.name + self.split + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, self.name + '_info.pkl')
        save_info(info_path, {'num_classes': self.num_classes})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.name + self.split + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, self.name + self.split + '_info.pkl')
        self.num_classes = load_info(info_path)['num_classes']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.name + self.split + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.name + self.split + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)