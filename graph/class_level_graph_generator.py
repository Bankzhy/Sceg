import csv
import json

import numpy as np

from graph.doc_sim import DocSim
from graph.matrix_construction import MatrixConstruction
import random

from graph.metrics_calculator import MetricsCalculator


class ClassLevelGraphGenerator:
    def __init__(self, sr_class, class_list):
        self.sr_class = sr_class
        self.class_list = class_list
        self.id_used = []
        self.nodes = []
        self.include_edges = []
        self.ssm_edges = []
        self.cdm_edges = []
        self.csm_edges = []
        self.max_code_length = 100000

    def __get_id(self):
        new_id = random.randint(0, self.max_code_length)
        if new_id in self.id_used:
            new_id = self.__get_id()
        else:
            self.id_used.append(new_id)
            return new_id

    def create_graph(self, target_method=None):
        metrics_calc = MetricsCalculator(
            sr_class=self.sr_class
        )

        nom = metrics_calc.get_nom()
        cis = metrics_calc.get_cis()
        noa = metrics_calc.get_noa()
        nopa = metrics_calc.get_nopa()
        atfd = metrics_calc.get_atfd(self.class_list)
        wmc = metrics_calc.get_wmc()
        tcc = metrics_calc.get_tcc()
        lcom = metrics_calc.get_lcom()
        dcc = metrics_calc.get_dcc(self.class_list)
        cam = metrics_calc.get_cam()
        dit = metrics_calc.get_dit(self.class_list)
        noam = metrics_calc.get_noam()
        doc_sim = DocSim()
        field_name_list = [f.field_name for f in self.sr_class.field_list]
        class_loc = metrics_calc.get_class_loc()
        dist = 0
        if target_method is not None:
            dist = metrics_calc.get_method_dist(target_method)

        new_class_node = {
            "id": self.__get_id(),
            "type": "class",
            "name": self.sr_class.class_name,
            "fields": ",".join(field_name_list),
            "metrics": {
                "nom": nom,
                "cis": cis,
                "noa": noa,
                "nopa": nopa,
                "atfd": atfd,
                "wmc": wmc,
                "tcc": tcc,
                "lcom": lcom,
                "dcc": dcc,
                "cam": cam,
                "dit": dit,
                "noam": noam,
                "class_loc": class_loc,
                "dist": dist
            }
        }
        self.nodes.append(new_class_node)
        method_nodes = []

        foreign_field_name_list = []
        foreign_method_name_list = []
        for cls in self.class_list:
            if cls.class_name == self.sr_class.class_name:
                continue

            for field in cls.field_list:
                foreign_field_name_list.append(field.field_name)
            for method in cls.method_list:
                foreign_method_name_list.append(method.method_name)


        for method in self.sr_class.method_list:
            loc = metrics_calc.get_method_loc(method)
            cc = metrics_calc.get_method_cc(method)
            pc = metrics_calc.get_method_pc(method)
            lcom1 = metrics_calc.get_method_LCOM1(method)
            lcom2 = metrics_calc.get_method_LCOM2(method)
            lcom3 = metrics_calc.get_method_LCOM3(method)
            lcom4 = metrics_calc.get_method_LCOM4(method)
            tsmc = metrics_calc.get_tsmc(method, doc_sim)
            nbd = metrics_calc.get_method_block_depth(method)
            fuc = metrics_calc.get_method_fuc(method)
            lmuc = metrics_calc.get_method_lmuc(method)
            noav = metrics_calc.get_method_noav(method)
            nfdi = metrics_calc.get_method_nfdi(method, foreign_method_name_list=foreign_method_name_list, foreign_field_name_list=foreign_field_name_list)
            nldi = metrics_calc.get_method_nldi(method)

            new_method_node = {
                'id': self.__get_id(),
                'type': "method",
                'name': method.method_name,
                "metrics": {
                    "loc": loc,
                    "cc": cc,
                    "pc": pc,
                    "lcom1": lcom1,
                    "lcom2": lcom2,
                    "lcom3": lcom3,
                    "lcom4": lcom4,
                    "tsmc": tsmc,
                    "nbd": nbd,
                    "fuc": fuc,
                    "lmuc": lmuc,
                    "noav": noav,
                    "nfdi": nfdi,
                    "nldi": nldi
                }
            }
            # self.nodes.append(new_method_node)
            method_nodes.append(new_method_node)

            new_include_edge = {
                "source": new_class_node["id"],
                "target": new_method_node["id"],
                "type": "include"
            }
            self.include_edges.append(new_include_edge)

        self.nodes.extend(method_nodes)

        mc = MatrixConstruction(self.sr_class)
        ssm, cdm, csm = mc.get_all_matrix()
        csm = mc.calculate_CSM_doc_sim(doc_sim=doc_sim)

        # ssm edges
        length = np.size(ssm, 0)
        for i in range(0, length):
            for j in range(0, length):
                if i == j:
                    continue
                if ssm[i][j] > 0:
                    new_ssm_edge = {
                        "source": method_nodes[i]["id"],
                        "target": method_nodes[j]["id"],
                        "type": "ssm"
                    }
                    self.ssm_edges.append(new_ssm_edge)

        # cdm edges
        length = np.size(cdm, 0)
        for i in range(0, length):
            for j in range(0, length):
                if i == j:
                    continue
                if cdm[i][j] > 0:
                    new_cdm_edge = {
                        "source": method_nodes[i]["id"],
                        "target": method_nodes[j]["id"],
                        "type": "cdm"
                    }
                    self.cdm_edges.append(new_cdm_edge)

        # csm edges
        length = np.size(csm, 0)
        for i in range(0, length):
            for j in range(0, length):
                if i == j:
                    continue
                if csm[i][j] > 0.5:
                    new_csm_edge = {
                        "source": method_nodes[i]["id"],
                        "target": method_nodes[j]["id"],
                        "type": "csm"
                    }
                    self.csm_edges.append(new_csm_edge)


    def create_merge_graph(self, source_class_graph, target_class_graph, doc_sim):
        source_class_node = None
        target_class_node = None
        merge_method_nodes = []
        for node in source_class_graph["nodes"]:
            if node["type"] == "class":
                source_class_node = node
            else:
                merge_method_nodes.append(node)
        for node in target_class_graph["nodes"]:
            if node["type"] == "class":
                target_class_node = node
            else:
                merge_method_nodes.append(node)
        metrics_calc = MetricsCalculator(
            sr_class=self.sr_class
        )
        merged_nom = source_class_node["metrics"]["nom"] + target_class_node["metrics"]["nom"]
        merged_cis = source_class_node["metrics"]["cis"] + target_class_node["metrics"]["cis"]
        merged_noa = source_class_node["metrics"]["noa"] + target_class_node["metrics"]["noa"]
        merged_nopa = source_class_node["metrics"]["nopa"] + target_class_node["metrics"]["nopa"]
        merged_atfd = source_class_node["metrics"]["atfd"] + target_class_node["metrics"]["atfd"]
        merged_wmc = source_class_node["metrics"]["wmc"] + target_class_node["metrics"]["wmc"]
        tcc = metrics_calc.get_tcc()
        lcom = metrics_calc.get_lcom()
        merged_dcc = source_class_node["metrics"]["dcc"] + target_class_node["metrics"]["dcc"]
        cam = metrics_calc.get_cam()
        merged_dit = max(source_class_node["metrics"]["dit"], target_class_node["metrics"]["dit"])
        merged_noam = source_class_node["metrics"]["noam"] + target_class_node["metrics"]["noam"]
        merged_class_loc = source_class_node["metrics"]["class_loc"] + target_class_node["metrics"]["class_loc"]

        new_class_node = {
            "id": self.__get_id(),
            "type": "class",
            "metrics": {
                "nom": merged_nom,
                "cis": merged_cis,
                "noa": merged_noa,
                "nopa": merged_nopa,
                "atfd": merged_atfd,
                "wmc": merged_wmc,
                "tcc": tcc,
                "lcom": lcom,
                "dcc": merged_dcc,
                "cam": cam,
                "dit": merged_dit,
                "noam": merged_noam,
                "class_loc": merged_class_loc,
                "dist": 0
            }
        }
        self.nodes.append(new_class_node)
        method_nodes = []
        for method in self.sr_class.method_list:
            loc = metrics_calc.get_method_loc(method)
            cc = metrics_calc.get_method_cc(method)
            pc = metrics_calc.get_method_pc(method)
            lcom1 = metrics_calc.get_method_LCOM1(method)
            lcom2 = metrics_calc.get_method_LCOM2(method)
            lcom3 = metrics_calc.get_method_LCOM3(method)
            lcom4 = metrics_calc.get_method_LCOM4(method)
            tsmc = metrics_calc.get_tsmc(method, doc_sim)
            nbd = metrics_calc.get_method_block_depth(method)
            fuc = metrics_calc.get_method_fuc(method)
            lmuc = metrics_calc.get_method_lmuc(method)
            noav = metrics_calc.get_method_noav(method)

            new_method_node = {
                'id': self.__get_id(),
                'type': "method",
                "metrics": {
                    "loc": loc,
                    "cc": cc,
                    "pc": pc,
                    "lcom1": lcom1,
                    "lcom2": lcom2,
                    "lcom3": lcom3,
                    "lcom4": lcom4,
                    "tsmc": tsmc,
                    "nbd": nbd,
                    "fuc": fuc,
                    "lmuc": lmuc,
                    "noav": noav
                }
            }
            # self.nodes.append(new_method_node)
            method_nodes.append(new_method_node)

            new_include_edge = {
                "source": new_class_node["id"],
                "target": new_method_node["id"],
                "type": "include"
            }
            self.include_edges.append(new_include_edge)

        self.nodes.extend(method_nodes)

        mc = MatrixConstruction(self.sr_class)
        ssm, cdm, csm = mc.get_all_matrix()
        doc_sim = DocSim()
        csm = mc.calculate_CSM_doc_sim(doc_sim=doc_sim)

        # ssm edges
        length = np.size(ssm, 0)
        for i in range(0, length):
            for j in range(0, length):
                if i == j:
                    continue
                if ssm[i][j] > 0:
                    new_ssm_edge = {
                        "source": method_nodes[i]["id"],
                        "target": method_nodes[j]["id"],
                        "type": "ssm"
                    }
                    self.ssm_edges.append(new_ssm_edge)

        # cdm edges
        length = np.size(cdm, 0)
        for i in range(0, length):
            for j in range(0, length):
                if i == j:
                    continue
                if cdm[i][j] > 0:
                    new_cdm_edge = {
                        "source": method_nodes[i]["id"],
                        "target": method_nodes[j]["id"],
                        "type": "cdm"
                    }
                    self.cdm_edges.append(new_cdm_edge)

        # csm edges
        length = np.size(csm, 0)
        for i in range(0, length):
            for j in range(0, length):
                if i == j:
                    continue
                if csm[i][j] > 0.5:
                    new_csm_edge = {
                        "source": method_nodes[i]["id"],
                        "target": method_nodes[j]["id"],
                        "type": "csm"
                    }
                    self.csm_edges.append(new_csm_edge)

    def fetch_class_node(self, sr_class, target_method):
        metrics_calc = MetricsCalculator(
            sr_class=sr_class
        )

        nom = metrics_calc.get_nom()
        cis = metrics_calc.get_cis()
        noa = metrics_calc.get_noa()
        nopa = metrics_calc.get_nopa()
        atfd = metrics_calc.get_atfd(self.class_list)
        wmc = metrics_calc.get_wmc()
        tcc = metrics_calc.get_tcc()
        lcom = metrics_calc.get_lcom()
        dcc = metrics_calc.get_dcc(self.class_list)
        cam = metrics_calc.get_cam()
        dit = metrics_calc.get_dit(self.class_list)
        noam = metrics_calc.get_noam()
        field_name_list = [f.field_name for f in sr_class.field_list]
        class_loc = metrics_calc.get_class_loc()
        dist = 0
        dist = metrics_calc.get_method_dist(target_method)

        new_class_node = {
            "id": self.__get_id(),
            "type": "class",
            "name": sr_class.class_name,
            "fields": ",".join(field_name_list),
            "metrics": {
                "nom": nom,
                "cis": cis,
                "noa": noa,
                "nopa": nopa,
                "atfd": atfd,
                "wmc": wmc,
                "tcc": tcc,
                "lcom": lcom,
                "dcc": dcc,
                "cam": cam,
                "dit": dit,
                "noam": noam,
                "class_loc": class_loc,
                "dist": dist
            }
        }
        return new_class_node
        # self.nodes.append(new_class_node)

    def fetch_method_node(self, sr_class, target_method):
        doc_sim = DocSim()
        metrics_calc = MetricsCalculator(
            sr_class=sr_class
        )
        foreign_field_name_list = []
        foreign_method_name_list = []
        for cls in self.class_list:
            if cls.class_name == sr_class.class_name:
                continue

            for field in cls.field_list:
                foreign_field_name_list.append(field.field_name)
            for method in cls.method_list:
                foreign_method_name_list.append(method.method_name)

        loc = metrics_calc.get_method_loc(target_method)
        cc = metrics_calc.get_method_cc(target_method)
        pc = metrics_calc.get_method_pc(target_method)
        lcom1 = metrics_calc.get_method_LCOM1(target_method)
        lcom2 = metrics_calc.get_method_LCOM2(target_method)
        lcom3 = metrics_calc.get_method_LCOM3(target_method)
        lcom4 = metrics_calc.get_method_LCOM4(target_method)
        tsmc = metrics_calc.get_tsmc(target_method, doc_sim)
        nbd = metrics_calc.get_method_block_depth(target_method)
        fuc = metrics_calc.get_method_fuc(target_method)
        lmuc = metrics_calc.get_method_lmuc(target_method)
        noav = metrics_calc.get_method_noav(target_method)
        nfdi = metrics_calc.get_method_nfdi(target_method, foreign_method_name_list=foreign_method_name_list,
                                            foreign_field_name_list=foreign_field_name_list)
        nldi = metrics_calc.get_method_nldi(target_method)

        new_method_node = {
            'id': self.__get_id(),
            'type': "method",
            'name': target_method.method_name,
            "metrics": {
                "loc": loc,
                "cc": cc,
                "pc": pc,
                "lcom1": lcom1,
                "lcom2": lcom2,
                "lcom3": lcom3,
                "lcom4": lcom4,
                "tsmc": tsmc,
                "nbd": nbd,
                "fuc": fuc,
                "lmuc": lmuc,
                "noav": noav,
                "nfdi": nfdi,
                "nldi": nldi
            }
        }
        return new_method_node

    def create_fe_graph(self, target_class, target_method):
        # self.create_graph(target_method=target_method)
        # self.sr_class = target_class
        # self.create_graph(target_method=target_method)
        source_class_node = self.fetch_class_node(sr_class=self.sr_class, target_method=target_method)
        target_class_node = self.fetch_class_node(sr_class=target_class, target_method=target_method)
        self.nodes.append(source_class_node)
        self.nodes.append(target_class_node)
        target_method_node = self.fetch_method_node(sr_class=target_class, target_method=target_method)
        self.nodes.append(target_method_node)





    def to_json(self):
        info = {}
        info["nodes"] = self.nodes
        info["include_edges"] = self.include_edges
        info["ssm_edges"] = self.ssm_edges
        info["cdm_edges"] = self.cdm_edges
        info["csm_edges"] = self.csm_edges
        return json.dumps(info)

    def to_dict(self):
        info = {}
        info["nodes"] = self.nodes
        info["include_edges"] = self.include_edges
        info["ssm_edges"] = self.ssm_edges
        info["cdm_edges"] = self.cdm_edges
        info["csm_edges"] = self.csm_edges
        return info

    def to_database(self, db, project_name, group, extract_methods=""):
        cursor = db.cursor()
        graph_json = self.to_json()
        path = self.sr_class.package_name + "."
        path = path + self.sr_class.class_name
        query = (r"replace into lc_master (project, class_name, content, extract_methods, `group`, split, graph, `path`, label, reviewer_id) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
        values = (project_name, self.sr_class.class_name, self.sr_class.class_name, extract_methods, group, "pool", graph_json, path, 9, 0)
        cursor.execute(query, values)
        db.commit()

    def to_fec_database(self, db, project_name, group, source_class_name, target_class_name, method_path, method_name):
        cursor = db.cursor()
        graph_json = self.to_json()
        path = self.sr_class.package_name + "."
        path = path + self.sr_class.class_name
        query = (r"replace into fe_master (project, class_name, method_name, target_class_name, `group`, split, graph, `path`, label, reviewer_id, content) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
        values = (project_name, source_class_name, method_name, target_class_name, group, "pool", graph_json, method_path, 9, 0, source_class_name)
        cursor.execute(query, values)
        db.commit()


    def to_list(self, db, project_name, group, extract_methods=""):
        graph_json = self.to_json()
        path = self.sr_class.package_name + "."
        path = path + self.sr_class.class_name
        return [project_name, self.sr_class.class_name, self.sr_class.class_name, extract_methods, group, "pool", graph_json, path, 9, 0]

    def to_fe_database(self, db, project_name, group, source_class_name, target_class_name, method_path, method_name):
        cursor = db.cursor()
        graph_json = self.to_json()
        method_path_l = method_path.split("_")
        mn = method_path_l[len(method_path_l)-1]
        query = (
            r"replace into fe_master (project, class_name, method_name, target_class_name, `group`, split, graph, `path`, label, reviewer_id, content) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
        values = (project_name, source_class_name, method_name, target_class_name, group, "pool", graph_json, method_path, 9, 0, source_class_name)
        cursor.execute(query, values)
        db.commit()