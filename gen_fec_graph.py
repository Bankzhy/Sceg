import csv
import json
import os
from pathlib import Path

import pymysql

from common import project_path_dict
from graph.class_level_graph_generator import ClassLevelGraphGenerator
from sitter.kast2core import KASTParse

db = pymysql.connect(
    host="47.113.220.80",
    user="root",
    password="Apple3328823%",
    database="sce",
    charset="utf8mb4",  # Use utf8mb4 for full Unicode support
    connect_timeout=500
)


def create_fec_graph(project, class_name):
    project_path = project_path_dict[project]
    ast = KASTParse(project_path, "java")
    ast.setup()
    sr_project = ast.do_parse()

    cls_list = []
    cls_name_list = []

    for program in sr_project.program_list:
        for sr_class in program.class_list:
            cls_list.append(sr_class)
            cls_name_list.append(sr_class.class_name)

    for program in sr_project.program_list:
        for sr_class in program.class_list:
            if sr_class.class_name == class_name:
                class_level_graph_generator = ClassLevelGraphGenerator(sr_class=sr_class, class_list=cls_list)
                class_level_graph_generator.create_graph()
                graph = class_level_graph_generator.to_dict()
                return graph
                # class_level_graph_generator.to_fec_database(db, project, group, class_name, target_class_name, "", method_name)

def create_auto_fec_graph(project, class_name, target_class_name, method_name):
    project_path = project_path_dict[project]
    ast = KASTParse(project_path, "java")
    ast.setup()
    sr_project = ast.do_parse()

    cls_list = []
    cls_name_list = []
    source_class = None
    target_class = None


    for program in sr_project.program_list:
        for sr_class in program.class_list:
            cls_list.append(sr_class)
            cls_name_list.append(sr_class.class_name)

    for program in sr_project.program_list:
        for sr_class in program.class_list:
            if sr_class.class_name == class_name:
                source_class = sr_class
            if sr_class.class_name == target_class_name:
                target_class = sr_class

            if source_class is not None and target_class is not None:
                for m in target_class.method_list:
                    if m.method_name == method_name:
                        source_class.method_list.append(m)
                class_level_graph_generator = ClassLevelGraphGenerator(sr_class=source_class, class_list=cls_list)
                class_level_graph_generator.create_graph()
                graph = class_level_graph_generator.to_dict()
                return graph

def load_graph_dict():
    cursor = db.cursor()
    query = "select * from fec_master;"
    cursor.execute(query)
    rows = cursor.fetchall()

    graph_dict = {}
    info_dict = {}

    for row in rows:
        fe_id = row[0]
        fe_project = row[1]
        fe_class_name = row[2]
        fe_method_name = row[3]
        fe_target_class_name = row[5]
        fe_group = row[6]
        fe_split = row[7]
        fe_graph = row[8]
        fe_graph = json.loads(fe_graph)
        fe_label = row[10]

        key = fe_project + "_" + fe_class_name

        new_info = [fe_project, fe_class_name, fe_method_name, fe_target_class_name, fe_group, fe_split, fe_label]
        info_dict[key] = new_info
        graph_dict[key] = fe_graph
    return graph_dict, info_dict


def check_exist_graph(project, class_name):
    cursor = db.cursor()
    query = "select * from fec_master where project=%s and class_name=%s;"
    cursor.execute(query, (project, class_name))
    row = cursor.fetchone()
    if row is not None:
        fe_graph = row[8]
        fe_graph = json.loads(fe_graph)
        return fe_graph

def update_fec(fec_graph, info, fe_id):
    cursor = db.cursor()
    key = info[0] + "_" + info[1]
    fe_graph = json.dumps(fec_graph)

    query = (
        r"replace into fec_master (project, class_name, method_name, target_class_name, `group`, split, graph, `path`, label, reviewer_id, content) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
    values = (info[0], info[1], info[2], info[3], info[4], info[5], fe_graph, key, info[6], 0, "")
    cursor.execute(query, values)

    query = (r"update fe_master set content='fec' where fe_id=%s")
    cursor.execute(query,(fe_id))

    db.commit()


def gen():
    cursor = db.cursor()
    # query = "select * from fe_master where content!='fec' and label!=9 limit 500"
    query = "select * from fe_master where content!='fec' and label=1 and `group`='a'"
    cursor.execute(query)
    rows = cursor.fetchall()

    # graph_dict = {}
    # info_dict = []
    # graph_dict, info_dict = load_graph_dict()
    exist_fe_ids = []
    # print("graph dict loaded")
    # print(info_dict)
    ignore =["Vec3d"]

    for row in rows:
        fe_id = row[0]
        print("parsing..", fe_id)
        fe_project = row[1]
        fe_class_name = row[2]

        if fe_class_name in ignore:
            continue

        fe_method_name = row[3]
        fe_target_class_name = row[5]
        fe_group = row[6]
        fe_split = row[7]
        fe_graph = row[8]
        fe_graph = json.loads(fe_graph)
        fe_label = row[10]

        fec_graph = check_exist_graph(fe_project, fe_class_name)
        info = [fe_project, fe_class_name, fe_method_name, fe_target_class_name, fe_group, fe_split,fe_label]
        # info_dict[key] = new_info

        if fec_graph is None:
            if fe_group == "a":
                fec_graph = create_auto_fec_graph(fe_project, fe_class_name, fe_target_class_name, fe_method_name)
            else:
                fec_graph = create_fec_graph(fe_project, fe_class_name)

            for node in fec_graph["nodes"]:
                if node["name"] == fe_method_name:
                    node["metrics"]["source_dist"] = fe_graph["nodes"][0]["metrics"]["dist"]
                    node["metrics"]["target_dist"] = fe_graph["nodes"][1]["metrics"]["dist"]
                    if fe_label == 1:
                        node["is_extract"] = 1
                    else:
                        node["is_extract"] = 0
                else:
                    node["metrics"]["source_dist"] = 0
                    node["metrics"]["target_dist"] = 0
                    node["is_extract"] = 0
            # graph_dict[key] = fec_graph
            update_fec(fec_graph, info, fe_id)
            print("add graph: ", fec_graph)
        else:
            for node in fec_graph["nodes"]:
                if node["name"] == fe_method_name:
                    node["metrics"]["source_dist"] = fe_graph["nodes"][0]["metrics"]["dist"]
                    node["metrics"]["target_dist"] = fe_graph["nodes"][1]["metrics"]["dist"]
                    if fe_label == 1:
                        node["is_extract"] = 1
                    update_fec(fec_graph, info, fe_id)
                    print("update graph: ", fec_graph)
                    break

        # exist_fe_ids.append(str(fe_id))

    # for key in graph_dict.keys():
    #     info = info_dict[key]
    #     fe_graph = json.dumps(graph_dict[key])
    #
    #     query = (
    #         r"replace into fec_master (project, class_name, method_name, target_class_name, `group`, split, graph, `path`, label, reviewer_id, content) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
    #     values = (info[0], info[1], info[2], info[3], info[4], info[5], fe_graph, key, info[6], 0, "")
    #     cursor.execute(query, values)
    #
    #     fe_id_str = ",".join(exist_fe_ids)
    #     query = (r"update fe_master set content='fec' where fe_id in ("+fe_id_str+ ")")
    #     cursor.execute(query)
    #
    #     db.commit()


if __name__ == '__main__':
    gen()