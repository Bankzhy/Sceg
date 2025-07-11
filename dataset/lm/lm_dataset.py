import json
import os
import random
from pathlib import Path

import torch
from dgl.data import DGLDataset
import dgl
import pandas as pd
import networkx as nx
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

class LMDataset(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：

        Parameters
        ----------
        url : str
            下载原始数据集的url。
        raw_dir : str
            指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
        save_dir : str
            处理完成的数据集的保存目录。默认：raw_dir指定的值
        force_reload : bool
            是否重新导入数据集。默认：False
        verbose : bool
            是否打印进度信息。
        """

    def __init__(self,
                 url=None,
                 split="train",
                 raw_dir=None,
                 force_reload=False,
                 verbose=False):
        self.graphs = []
        self.labels = []
        self.split = split
        super(LMDataset, self).__init__(name='lm_dataset',
                                        url=url,
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # 将原始数据下载到本地磁盘
        pass

    def process(self):
        # 将原始数据处理为图、标签和数据集划分的掩码
        self.graphs, self.labels = self._load_graph()

    def _load_graph(self):
        graphs = []
        labels = []

        pos_path = Path("dataset/lm/"+self.split + "_1.txt")
        neg_path = Path("dataset/lm/"+self.split + "_0.txt")
        failed_num = 0

        with open(pos_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                new_graph = json.loads(line)
                new_graph = self._get_graph(new_graph)
                graphs.append(new_graph)
                labels.append(1)

        with open(neg_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                new_graph = json.loads(line)
                new_graph = self._get_graph(new_graph)
                graphs.append(new_graph)
                labels.append(0)

        labels = torch.tensor(labels)
        self.num_classes = 2
        print("Failed:", failed_num)
        return graphs, labels

    def _get_graph(self, lm_graph):
        nodes = lm_graph["nodes"]
        statement_nodes_ids = []
        method_nodes_ids = []
        cf_edges = lm_graph["flow_edges"]
        cd_edges = lm_graph["cd_edges"]
        dd_edges = lm_graph["dd_edges"]
        include_edges = lm_graph["include_edges"]

        method_feature = []
        nodes_feature = []
        cf_edge_index = [[],[]]
        dd_edge_index = [[],[]]
        cd_edge_index = [[],[]]
        include_edge_index = [[],[]]

        for node in nodes:
            if "type" in node.keys():
                feature = []
                feature.append(node["metrics"]["loc"])
                feature.append(node["metrics"]["cc"])
                feature.append(node["metrics"]["pc"])
                feature.append(node["metrics"]["lcom1"])
                feature.append(node["metrics"]["lcom2"])
                feature.append(node["metrics"]["lcom3"])
                feature.append(node["metrics"]["lcom4"])
                feature.append(node["metrics"]["noav"])
                method_feature.append(feature)
                method_nodes_ids.append(node["id"])
            else:
                feature = []
                feature.append(node["metrics"]["abcl"])
                feature.append(node["metrics"]["fuc"])
                feature.append(node["metrics"]["lmuc"])
                feature.append(node["metrics"]["puc"])
                feature.append(node["metrics"]["nbd"])
                feature.append(node["metrics"]["vuc"])
                feature.append(node["metrics"]["wc"])
                feature.append(node["metrics"]["tsmm"])
                nodes_feature.append(feature)
                statement_nodes_ids.append(node["id"])

        if len(nodes_feature) <= 0:
            nodes_feature.append([0]*len(method_feature[0]))

        for edge in cf_edges:
            source = statement_nodes_ids.index(edge["source"])
            target = statement_nodes_ids.index(edge["target"])
            cf_edge_index[0].append(source)
            cf_edge_index[1].append(target)

        for edge in cd_edges:
            source = statement_nodes_ids.index(edge["source"])
            target = statement_nodes_ids.index(edge["target"])
            cd_edge_index[0].append(source)
            cd_edge_index[1].append(target)

        for edge in dd_edges:
            source = statement_nodes_ids.index(edge["source"])
            target = statement_nodes_ids.index(edge["target"])
            dd_edge_index[0].append(source)
            dd_edge_index[1].append(target)

        for edge in include_edges:
            source = method_nodes_ids.index(edge["source"])
            target = statement_nodes_ids.index(edge["target"])
            include_edge_index[0].append(source)
            include_edge_index[1].append(target)

        data_dict = {}

        if len(cf_edge_index) > 0 and len(cf_edge_index[0]) > 0:
            data_dict[('statement', 'cf', 'statement')] = (torch.tensor(cf_edge_index[0]), torch.tensor(cf_edge_index[1]))
        else:
            data_dict[('statement', 'cf', 'statement')] = (torch.tensor([0]), torch.tensor([0]))
            # pass


        if len(dd_edge_index) > 0 and len(dd_edge_index[0]) > 0:
            data_dict[('statement', 'dd', 'statement')] = (
            torch.tensor(dd_edge_index[0]), torch.tensor(dd_edge_index[1]))
        else:
            data_dict[('statement', 'dd', 'statement')] = (
                torch.tensor([0]), torch.tensor([0]))
            # pass

        if len(cd_edge_index) > 0 and len(cd_edge_index[0]) > 0:
            data_dict[('statement', 'cd', 'statement')] = (
            torch.tensor(cd_edge_index[0]), torch.tensor(cd_edge_index[1]))
        else:
            data_dict[('statement', 'cd', 'statement')] = (
                torch.tensor([0]), torch.tensor([0]))
            # pass

        if len(include_edge_index) > 0 and len(include_edge_index[0]) > 0:
            data_dict[('method', 'include', 'statement')] = (torch.tensor(include_edge_index[0]), torch.tensor(include_edge_index[1]))
        else:
            data_dict[('method', 'include', 'statement')] = (torch.tensor([0]), torch.tensor([0]))
            # pass

        num_nodes_dict = {'statement': len(nodes_feature)}
        num_nodes_dict['method'] = 1

        G = dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)

        G.nodes['method'].data['feat'] = torch.tensor(method_feature).to(torch.float32)
        G.nodes['statement'].data['feat'] = torch.tensor(nodes_feature).to(torch.float32)

        # node cls
        # G.nodes['statement'].data['label'] = torch.tensor(nodes_labels)

        print(G)

        return G

    # def _get_graph(self, new_graph):
    #     nodes_ids = []
    #     cf_edge_index = []
    #     dd_edge_index = []
    #     cd_edge_index = []
    #     nodes_feature = []
    #     nodes_labels = []
    #     method_feature = []
    #
    #
    #     # add node feature
    #     node_df = pd.read_csv(path / "node.csv")
    #     nodes_ids = list(node_df['id'])
    #     # nodes_labels = list(node_df['from_cps'])
    #     # nodes_feature = list(node_df['category'])
    #     node_row_num = node_df.shape[0]
    #     for i in range(0, node_row_num):
    #         # nodes_feature.append(list(node_df.iloc[i, 1:(node_df.shape[1]-1)]))
    #         nodes_feature.append(list(node_df.iloc[i, 1:]))
    #
    #
    #
    #     num_nodes_dict = {'statement': len(nodes_ids)}
    #     num_nodes_dict['method'] = 1
    #
    #
    #     # add method feature
    #     method_df = pd.read_csv(path / "method.csv")
    #     method_row_num = method_df.shape[0]
    #     for i in range(0, method_row_num):
    #         method_feature.append(list(method_df.iloc[i, 1:]))
    #
    #     # add cf_edge_index
    #     cf_edge_index_df = pd.read_csv(path / "cf_edge_index.csv")
    #     edge_index_row_num = cf_edge_index_df.shape[0]
    #     for i in range(0, edge_index_row_num):
    #         f = lambda x: nodes_ids.index(x)
    #         cf_edge_index.append(list(cf_edge_index_df.iloc[i, 1:].apply(f)))
    #
    #     # add dd_edge_index
    #     dd_edge_index_df = pd.read_csv(path / "dd_edge_index.csv")
    #     edge_index_row_num = dd_edge_index_df.shape[0]
    #     for i in range(0, edge_index_row_num):
    #         f = lambda x: nodes_ids.index(x)
    #         dd_edge_index.append(list(dd_edge_index_df.iloc[i, 1:].apply(f)))
    #
    #     # add cd_edge_index
    #     cd_edge_index_df = pd.read_csv(path / "cd_edge_index.csv")
    #     edge_index_row_num = cd_edge_index_df.shape[0]
    #     for i in range(0, edge_index_row_num):
    #         f = lambda x: nodes_ids.index(x)
    #         cd_edge_index.append(list(cd_edge_index_df.iloc[i, 1:].apply(f)))
    #
    #     data_dict = {}
    #
    #
    #
    #     if len(cf_edge_index) > 0 and len(cf_edge_index[0]) > 0:
    #         data_dict[('statement', 'cf', 'statement')] = (torch.tensor(cf_edge_index[0]), torch.tensor(cf_edge_index[1]))
    #     else:
    #         data_dict[('statement', 'cf', 'statement')] = (torch.tensor([0]), torch.tensor([0]))
    #     if len(dd_edge_index) > 0 and len(dd_edge_index[0]) > 0:
    #         data_dict[('statement', 'dd', 'statement')] = (
    #         torch.tensor(dd_edge_index[0]), torch.tensor(dd_edge_index[1]))
    #     else:
    #         data_dict[('statement', 'dd', 'statement')] = (
    #             torch.tensor([0]), torch.tensor([0]))
    #     if len(cd_edge_index) > 0 and len(cd_edge_index[0]) > 0:
    #         data_dict[('statement', 'cd', 'statement')] = (
    #         torch.tensor(cd_edge_index[0]), torch.tensor(cd_edge_index[1]))
    #     else:
    #         data_dict[('statement', 'cd', 'statement')] = (
    #             torch.tensor([0]), torch.tensor([0]))
    #
    #     method_nodes = []
    #     statement_nodes = []
    #     for index, id in enumerate(nodes_ids):
    #         method_nodes.append(0)
    #         statement_nodes.append(index)
    #
    #     data_dict[('method', 'include', 'method')] = (torch.tensor([0]), torch.tensor([0]))
    #     data_dict[('method', 'include', 'statement')] = (torch.tensor(method_nodes), torch.tensor(statement_nodes))
    #
    #     G = dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)
    #
    #     G.nodes['method'].data['feat'] = torch.tensor(method_feature).to(torch.float32)
    #     G.nodes['statement'].data['feat'] = torch.tensor(nodes_feature).to(torch.float32)
    #
    #     # node cls
    #     # G.nodes['statement'].data['label'] = torch.tensor(nodes_labels)
    #
    #     print(G)
    #
    #     return G

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 2


    def __getitem__(self, idx):
        # 通过idx得到与之对应的一个样本
        return self.graphs[idx], self.labels[idx]

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