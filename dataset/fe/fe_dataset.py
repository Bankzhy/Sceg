import os
import dgl
import json
import torch
from pathlib import Path
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info

class FEDataset(DGLDataset):
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
        self.class_list = []
        self.method_list = []
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
        self.graphs, self.labels = self._load_graph()

    def _load_graph(self):
        graphs = []
        labels = []

        pos_path = Path("dataset/fe/"+self.split + "_1.txt")
        neg_path = Path("dataset/fe/"+self.split + "_0.txt")
        failed_num = 0

        with open(pos_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # try:
                    new_graph = json.loads(line)
                    new_graph, pos_nodes = self._get_graph(new_graph)

                    if self.split == 'train':
                        if pos_nodes > 0:
                            graphs.append(new_graph)
                            labels.append(1)
                        else:
                            failed_num += 1
                    else:
                        graphs.append(new_graph)
                        labels.append(1)
                # except Exception as e:
                #     failed_num += 1
                #     print(e)
                #     continue

        with open(neg_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # try:
                    new_graph = json.loads(line)
                    new_graph, pos_nodes = self._get_graph(new_graph)

                    if self.split == 'train':
                        if pos_nodes > 0:
                            graphs.append(new_graph)
                            labels.append(1)
                        else:
                            failed_num += 1
                    else:
                        graphs.append(new_graph)
                        labels.append(1)

                    # if pos_nodes > 0:
                    #     graphs.append(new_graph)
                    #     labels.append(0)
                    # else:
                    #     failed_num += 1
                # except Exception as e:
                #     failed_num += 1
                #     print(e)
                #     continue


        labels = torch.tensor(labels)
        self.num_classes = 2
        print("Failed:", failed_num)
        return graphs, labels

    def _get_graph(self, lc_graph):
        nodes = lc_graph["nodes"]
        method_nodes_ids = []
        class_nodes_ids = []
        ssm_edges = lc_graph["ssm_edges"]
        cdm_edges = lc_graph["cdm_edges"]
        csm_edges = lc_graph["csm_edges"]
        include_edges = lc_graph["include_edges"]
        nodes_labels = []

        class_feature = []
        nodes_feature = []
        ssm_edge_index = [[],[]]
        cdm_edge_index = [[],[]]
        csm_edge_index = [[],[]]
        include_edge_index = [[],[]]

        pos_nodes = 0

        method_list = []

        for index, node in enumerate(nodes):
            if index == 0:
                feature = []
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
                feature.append(node["metrics"]["class_loc"])
                class_feature.append(feature)
                class_nodes_ids.append(node["id"])
                self.class_list.append(node['name'])
            else:
                feature = []
                # feature.append(node["metrics"]["loc"])
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
                feature.append(node["metrics"]["source_dist"])
                feature.append(node["metrics"]["target_dist"])
                nodes_feature.append(feature)
                method_nodes_ids.append(node["id"])
                nodes_labels.append(node["is_extract"])
                if node["is_extract"] == 1:
                    pos_nodes += 1
                method_list.append(node['name'])
        self.method_list.append(method_list)


        if len(nodes_feature) <= 0:
            nodes_feature.append([0]*len(class_feature[0]))
            nodes_labels.append(0)

        for edge in ssm_edges:
            source = method_nodes_ids.index(edge["source"])
            target = method_nodes_ids.index(edge["target"])
            ssm_edge_index[0].append(source)
            ssm_edge_index[1].append(target)

        for edge in cdm_edges:
            source = method_nodes_ids.index(edge["source"])
            target = method_nodes_ids.index(edge["target"])
            cdm_edge_index[0].append(source)
            cdm_edge_index[1].append(target)

        for edge in csm_edges:
            source = method_nodes_ids.index(edge["source"])
            target = method_nodes_ids.index(edge["target"])
            csm_edge_index[0].append(source)
            csm_edge_index[1].append(target)

        for edge in include_edges:
            source = class_nodes_ids.index(edge["source"])
            target = method_nodes_ids.index(edge["target"])
            include_edge_index[0].append(source)
            include_edge_index[1].append(target)

        data_dict = {}

        if len(ssm_edge_index) > 0 and len(ssm_edge_index[0]) > 0:
            data_dict[('method', 'ssm', 'method')] = (torch.tensor(ssm_edge_index[0]), torch.tensor(ssm_edge_index[1]))
        else:
            data_dict[('method', 'ssm', 'method')] = (torch.tensor([0]), torch.tensor([0]))
            # pass


        if len(cdm_edge_index) > 0 and len(cdm_edge_index[0]) > 0:
            data_dict[('method', 'cdm', 'method')] = (
            torch.tensor(cdm_edge_index[0]), torch.tensor(cdm_edge_index[1]))
        else:
            data_dict[('method', 'cdm', 'method')] = (
                torch.tensor([0]), torch.tensor([0]))
            # pass

        if len(csm_edge_index) > 0 and len(csm_edge_index[0]) > 0:
            data_dict[('method', 'csm', 'method')] = (
            torch.tensor(csm_edge_index[0]), torch.tensor(csm_edge_index[1]))
        else:
            data_dict[('method', 'csm', 'method')] = (
                torch.tensor([0]), torch.tensor([0]))
            # pass

        if len(include_edge_index) > 0 and len(include_edge_index[0]) > 0:
            data_dict[('class', 'include', 'method')] = (torch.tensor(include_edge_index[0]), torch.tensor(include_edge_index[1]))
        else:
            data_dict[('class', 'include', 'method')] = (torch.tensor([0]), torch.tensor([0]))
            # pass

        num_nodes_dict = {'method': len(nodes_feature)}
        num_nodes_dict['class'] = 1
        data_dict[('class', 'include', 'class')] = (torch.tensor([0]), torch.tensor([0]))

        G = dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)

        G.nodes['class'].data['feat'] = torch.tensor(class_feature).to(torch.float32)
        G.nodes['method'].data['feat'] = torch.tensor(nodes_feature).to(torch.float32)

        # node cls
        # G.nodes['statement'].data['label'] = torch.tensor(nodes_labels)
        G.nodes['method'].data['label'] = torch.tensor(nodes_labels)

        print(G)

        return G, pos_nodes

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