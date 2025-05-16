class Graph:
    def __init__(self, node_list, edge_list):
        self.node_list = node_list
        self.edge_list = edge_list

    def incoming_edges_of(self, node):
        result = []
        for edge in self.edge_list:
            if edge.target == node.id:
                result.append(edge)
        return result

    def outgoing_edges_of(self, node_id):
        result = []
        for edge in self.edge_list:
            if edge.source == node_id:
                result.append(edge)
        return result

    def get_edge_source(self, edge):
        id = edge.source
        node = None
        for n in self.node_list:
            if n.id == id:
                node = n
        return node

    def get_edge_target(self, edge):
        id = edge.target
        node = None
        for n in self.node_list:
            if n.id == id:
                node = n
        return node

    def get_edge(self, source, target):
        edge = None
        for e in self.edge_list:
            if e.source == source and e.target == target:
                edge = e
        return edge

    def dfs_node_list(self, node_id):
        result = []
        result.append(node_id)
        out_edges = self.outgoing_edges_of(node_id)
        for edge in out_edges:
            target_node = edge.target
            t_e = []
            t_e = self.dfs_node_list(target_node)
            if len(t_e) > 0:
                for t in t_e:
                    if t not in result:
                        result.append(t)
        return result

    def get_node(self, id):
        for node in self.node_list:
            if node.id == id:
                return node

    def get_dfs_node_list(self):
        result = []
        dfs = []
        for node in self.node_list:
            if node.category == 4:
                dfs = self.dfs_node_list(node.id)
        for d in dfs:
            result.append(self.get_node(d))
        return result


class Node:
    def __init__(self, id):
        self.id = id


class Edge:
    def __init__(self, id, source, target):
        self.id = id
        self.source = source
        self.target = target