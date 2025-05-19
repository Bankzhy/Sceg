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

    def create_graph(self):
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

        new_class_node = {
            "id": self.__get_id(),
            "type": "class",
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
                "noam": noam
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


    def create_merge_graph(self, source_class_graph, target_class_graph):



    def to_json(self):
        info = {}
        info["nodes"] = self.nodes
        info["include_edges"] = self.include_edges
        info["ssm_edges"] = self.ssm_edges
        info["cdm_edges"] = self.cdm_edges
        info["csm_edges"] = self.csm_edges
        return json.dumps(info)

    def to_database(self, db, project_name, group):
        cursor = db.cursor()
        graph_json = self.to_json()
        query = (r"replace into lc_master (project, class_name, content, extract_methods, `group`, split, graph, `path`, label, reviewer_id) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
        values = (project_name, self.sr_class.class_name, self.sr_class.class_name, "", group, "pool", graph_json, self.sr_class.package_name, 9, 0)
        cursor.execute(query, values)
        db.commit()