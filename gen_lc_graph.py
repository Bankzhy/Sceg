import csv
OVER_SIZE_LIMIT = 200_000_000

csv.field_size_limit(OVER_SIZE_LIMIT)

# 以下、書きたい処理
import json
import os
from pathlib import Path

import pymysql

from common import project_auto_dict
from gen_merged import project_path_dict
from graph.class_level_graph_generator import ClassLevelGraphGenerator
from graph.doc_sim import DocSim
from graph.matrix_construction import MatrixConstruction
from graph.pdg_generator import PDGGenerator
from sitter.kast2core import KASTParse

db = pymysql.connect(
    host="47.113.220.80",
    user="root",
    password="Apple3328823%",
    database="sce",
    charset="utf8mb4",  # Use utf8mb4 for full Unicode support
    connect_timeout=5000
)

def gen_original_graph(project_name):
    project_path = project_path_dict[project_name]
    ast = KASTParse(project_path, "java")
    ast.setup()
    sr_project = ast.do_parse()
    class_list = []
    for program in sr_project.program_list:
        for sr_class in program.class_list:
            class_list.append(sr_class)

    for program in sr_project.program_list:
        for sr_class in program.class_list:
            if sr_class.class_name == "Flowable" or sr_class.class_name == "Observable":
                continue
            class_level_graph_generator=ClassLevelGraphGenerator(sr_class=sr_class, class_list=class_list)
            print(sr_class.class_name)
            class_level_graph_generator.create_graph()
            class_level_graph_generator.to_database(db=db, project_name=project_name, group="original")


def find_class_graph_from_database(name, project):
    cursor = db.cursor()
    query = "select `graph` from lc_master where `class_name` ='" + str(name) + "' and project = '" + str(project) + "';"
    cursor.execute(query)
    data = cursor.fetchall()
    print(data)
    graph = data[0][0]
    graph = json.loads(graph)

    return graph

def gen_auto_graph(project_name):
    path = project_auto_dict[project_name] / "lc/"
    doc_sim = DocSim()
    csv_rows = []

    with open(path / "index.csv", mode='r') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            file_path = path / row[0]
            ast = KASTParse(file_path, "java")
            ast.setup()
            file = open(file_path, encoding='utf-8')
            file_content = file.read()
            sr_project = ast.do_parse_content(file_content)
            for program in sr_project.program_list:
                for sr_class in program.class_list:
                    try:
                        merged_class_name = sr_class.class_name
                        merged_class_name_l = merged_class_name.split("_")
                        target_class_name = merged_class_name_l[2]
                        source_class_name = merged_class_name_l[3]

                        if source_class_name == "Flowable" or source_class_name == "Observable":
                            continue
                        if target_class_name == "Flowable" or target_class_name == "Observable":
                            continue

                        source_class_graph = find_class_graph_from_database(source_class_name, project_name)
                        target_class_graph = find_class_graph_from_database(target_class_name, project_name)

                        class_level_graph_generator = ClassLevelGraphGenerator(sr_class=sr_class, class_list=[])
                        class_level_graph_generator.create_merge_graph(source_class_graph=source_class_graph, target_class_graph=target_class_graph, doc_sim=doc_sim)
                        # print(row[4])
                        # class_level_graph_generator.to_database(db=db, project_name=project_name, group="auto", extract_methods=row[4])

                        row = class_level_graph_generator.to_list(db=db, project_name=project_name, group="auto", extract_methods=row[4])
                        csv_rows.append(row)
                    except Exception as e:
                        print(e)
                        continue

    file_order = ["project", "class_name", "content", "extract_methods", "group", "split", "graph",
                  "path", "label", "reviewer_id"]
    with open((project_name+"_index.csv"), "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, file_order)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(dict(zip(file_order, row)))





def from_csv():
    cursor = db.cursor()

    with open('grootindex.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            print(row)
            try:
                query = (
                    r"replace into lc_master (project, class_name, content, extract_methods, `group`, split, graph, `path`, label, reviewer_id) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
                values = (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
                cursor.execute(query, values)
                db.commit()
            except Exception as e:
                print(e)
                continue


def mark_pos_nodes():
    cursor = db.cursor()
    cursor.execute("SELECT * FROM lc_master where `label`=1 and split='eval'")
    failed = 0
    for row in cursor.fetchall():
        try:
            lc_id = row[0]
            lc_graph = row[7]
            lc_graph = json.loads(lc_graph)
            extract_methods = row[4]
            extract_methods = extract_methods.split(",")
            group = row[5]

            if group == "a":
                project_name = row[1]
                merged_class_name = row[2]
                merged_class_name_l = merged_class_name.split("_")
                target_class_name = merged_class_name_l[2]
                source_class_name = merged_class_name_l[3]
                source_class_graph = find_class_graph_from_database(source_class_name, project_name)
                target_class_graph = find_class_graph_from_database(target_class_name, project_name)

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

                nodes = lc_graph["nodes"]
                for index, node in enumerate(nodes):
                    if index == 0:
                        continue

                    if merge_method_nodes[index-1] in extract_methods:
                        node["name"] = merge_method_nodes[index-1]["name"]
                        node["is_extract"] = 1
                    else:
                        node["name"] = merge_method_nodes[index - 1]["name"]
                        node["is_extract"] = 0
                lc_graph_str = json.dumps(lc_graph)
                query = (r"update lc_master set graph=%s where lc_id=%s;")
                values = (lc_graph_str, lc_id)
                print(query)
                cursor.execute(query, values)

            else:
                nodes = lc_graph["nodes"]
                for index, node in enumerate(nodes):
                    if index == 0:
                        continue

                    if node["name"] in extract_methods:
                        node["is_extract"] = 1
                    else:
                        node["is_extract"] = 0
                lc_graph_str = json.dumps(lc_graph)
                query = (r"update lc_master set graph=%s where lc_id=%s;")
                values = (lc_graph_str, lc_id)
                print(query)
                cursor.execute(query, values)
        except Exception as e:
            print(e)
            failed += 1

    db.commit()
    print(failed)
    print(nodes)



def fix_auto_graph():
    cursor = db.cursor()
    cursor.execute("select * from lc_master where split='train' and graph not like '%fields%';")
    for row in cursor.fetchall():
        lc_id = row[0]
        project_name = row[1]
        lc_class_name = row[2]
        lc_graph = row[7]
        lc_graph = json.loads(lc_graph)

        merged_class_name = lc_class_name
        merged_class_name_l = merged_class_name.split("_")
        target_class_name = merged_class_name_l[2]
        source_class_name = merged_class_name_l[3]

        if source_class_name == "Flowable" or source_class_name == "Observable":
            continue
        if target_class_name == "Flowable" or target_class_name == "Observable":
            continue

        source_class_graph = find_class_graph_from_database(source_class_name, project_name)
        target_class_graph = find_class_graph_from_database(target_class_name, project_name)

        merged_fields = source_class_graph['nodes'][0]["fields"] + "," + target_class_graph["nodes"][0]["fields"]
        lc_graph["nodes"][0]["fields"] = merged_fields
        lc_graph_str = json.dumps(lc_graph)

        print(merged_fields)
        # print(lc_graph_str)
        print("=====================")

        query = (r"update lc_master set graph=%s where lc_id=%s;")
        values = (lc_graph_str, lc_id)
        print(query)
        cursor.execute(query, values)

    db.commit()





if __name__ == '__main__':
    # for index, key in enumerate(project_path_dict.keys()):
    #     print("=================================")
    #     print(key)
    #     gen_original_graph(key)
    # for index, key in enumerate(project_auto_dict.keys()):
    #     if index == 0:
    #         continue
    #     if index == 1:
    #         continue
    #     if index == 9:
    #         continue
    #     print("=================================")
    #     print(key)
    #     print("=================================")
    #     gen_auto_graph(key)
    # gen_auto_graph("rxJava")
    # from_csv()
    # mark_pos_nodes()
    fix_auto_graph()