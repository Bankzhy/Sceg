import json
import random


class CIDGGenerator:
    def __init__(self, sr_class):
        self.sr_class = sr_class
        self.node_list = []
        self.field_node_list = []
        self.method_node_list = []
        self.fm_edge_list = []
        self.mm_edge_list = []
        self.cn_edge_list = []
        self.edge_list = []
        self.id_used = []
        self.max_code_length = 100000

    def __create_class_node(self):
        class_node = CIDGNodeClass(
            id=self.__get_id(),
            sr_class=self.sr_class
        )
        for node in self.node_list:
            new_cn_edge = CIDGEdgeCN(
                source=class_node.id,
                target=node.id,
                id=self.__get_id()
            )
            self.cn_edge_list.append(new_cn_edge)
        self.edge_list.extend(self.cn_edge_list)
        self.node_list.append(class_node)

    def create_graph(self):
        self.__create_node_list()
        self.__create_mm_edge()
        self.__create_fm_edge()
        self.__create_ff_edge()
        self.__create_class_node()

    def __create_node_list(self):
        for field in self.sr_class.field_list:
            new_node = CIDGNodeField(
                id=self.__get_id(),
                sr_field=field
            )
            self.field_node_list.append(new_node)

        for method in self.sr_class.method_list:
            new_node = CIDGNodeMethod(
                id=self.__get_id(),
                sr_method=method
            )
            self.method_node_list.append(new_node)

        self.node_list.extend(self.field_node_list)
        self.node_list.extend(self.method_node_list)

    def __create_fm_edge(self):
        for mn in self.method_node_list:
            for fn in self.field_node_list:
                if mn.sr_method.find_keyword(keyword=fn.sr_field.field_name):
                    new_edge = CIDGEdgeFM(
                        id=self.__get_id(),
                        source=mn.id,
                        target=fn.id
                    )
                    self.fm_edge_list.append(new_edge)
        self.edge_list.extend(self.fm_edge_list)

    def __create_mm_edge(self):
        for mn1 in self.method_node_list:
            for mn2 in self.method_node_list:
                if mn1.id != mn2.id:
                    if mn2.sr_method.find_keyword(keyword=mn1.sr_method.method_name):
                        new_edge = CIDGEdgeMM(
                            id=self.__get_id(),
                            source=mn2.id,
                            target=mn1.id
                        )
                        self.mm_edge_list.append(new_edge)
        self.edge_list.extend(self.mm_edge_list)

    def __create_ff_edge(self):
        pass

    def to_json(self):
        info={}
        info['nodes'] = []
        info['edges'] = []
        for node in self.node_list:
            info['nodes'].append(node.to_dic())

        for edge in self.edge_list:
            info['edges'].append(edge.to_dic())
        return json.dumps(info)

    def get_csv(self):
        pass

    def __get_id(self):
        new_id = random.randint(0, self.max_code_length)
        if new_id in self.id_used:
            new_id = self.__get_id()
        else:
            self.id_used.append(new_id)
            return new_id


class CIDGNode:

    def __init__(self, id):
        self.id = id

    def to_json(self):
        info = {}
        info['id'] = self.id
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info['id'] = self.id
        return info


class CIDGNodeClass(CIDGNode):
    def __init__(self, id, sr_class):
        self.id = id
        self.sr_class = sr_class

    def to_json(self):
        info = {}
        info['id'] = self.id
        info['type'] = "class"
        info['className'] = self.sr_class.class_name
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info['id'] = self.id
        info['type'] = "class"
        info['className'] = self.sr_class.class_name
        return info


class CIDGNodeField(CIDGNode):
    def __init__(self, id, sr_field):
        self.id = id
        self.sr_field = sr_field

    def to_json(self):
        info = {}
        info['id'] = self.id
        info['type'] = "field"
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info['id'] = self.id
        info['type'] = "field"
        info['fieldType'] = self.sr_field.field_type
        info['fieldName'] = self.sr_field.field_name
        info['fieldValue'] = self.sr_field.field_value
        info['modifiers'] = self.sr_field.modifiers

        return info


class CIDGNodeMethod(CIDGNode):
    def __init__(self, id, sr_method):
        self.id = id
        self.sr_method = sr_method

    def to_json(self):
        info = {}
        info['id'] = self.id
        info['type'] = "method"
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info['id'] = self.id
        info['methodId'] = self.sr_method.id
        info['type'] = "method"
        info["methodName"] = self.sr_method.method_name
        info["returnType"] = self.sr_method.return_type
        info["modifiers"] = self.sr_method.modifiers
        return info


class CIDGEdge:
    def __init__(self, id, source, target):
        self.id = id
        self.source = source
        self.target = target

    def to_json(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        return info


class CIDGEdgeFF(CIDGEdge):
    def __init__(self, id, source, target):
        self.id = id
        self.source = source
        self.target = target

    def to_json(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        info['type'] = "ff"
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        info['type'] = "ff"
        return info


class CIDGEdgeMM(CIDGEdge):
    def __init__(self, id, source, target):
        self.id = id
        self.source = source
        self.target = target

    def to_json(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        info['type'] = "mm"
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        info['type'] = "mm"
        return info


class CIDGEdgeFM(CIDGEdge):
    def __init__(self, id, source, target):
        self.id = id
        self.source = source
        self.target = target

    def to_json(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        info['type'] = "fm"
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        info['type'] = "fm"
        return info


class CIDGEdgeCN(CIDGEdge):
    def __init__(self, id, source, target):
        self.id = id
        self.source = source
        self.target = target

    def to_json(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        info['type'] = "cn"
        return json.dumps(info)

    def to_dic(self):
        info = {}
        info['id'] = self.id
        info['source'] = self.source
        info['target'] = self.target
        info['type'] = "cn"
        return info