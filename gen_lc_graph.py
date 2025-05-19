import csv
import json
import os
from pathlib import Path

import pymysql

from gen_merged import project_path_dict
from graph import doc_sim
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
    connect_timeout=50
)

project_path_dict = {
    "jgrapht": Path(r"D:\research\code_corpus\jgrapht")
}
project_auto_dict = {
    "jgrapht": Path(r"D:\research\code_corpus\jgrapht_auto")
}

def gen_original_graph(project_name):
    project_path = project_path_dict[project_name]
    ast = KASTParse(project_path, "java")
    ast.setup()
    sr_project = ast.do_parse()
    for program in sr_project.program_list:
        for sr_class in program.class_list:
            class_level_graph_generator=ClassLevelGraphGenerator(sr_class=sr_class, class_list=program.class_list)
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

    for file in os.listdir(path):
        file_path = path / file
        ast = KASTParse(file_path, "java")
        ast.setup()
        file = open(file_path, encoding='utf-8')
        file_content = file.read()
        sr_project = ast.do_parse_content(file_content)
        for program in sr_project.program_list:
            for sr_class in program.class_list:
                merged_class_name = sr_class.class_name
                merged_class_name_l = merged_class_name.split("_")
                target_class_name = merged_class_name_l[2]
                source_class_name = merged_class_name[3]

                source_class_graph = find_class_graph_from_database(source_class_name, project_name)
                target_class_graph = find_class_graph_from_database(target_class_name, project_name)

                class_level_graph_generator = ClassLevelGraphGenerator(sr_class=sr_class)
                class_level_graph_generator.create_merge_graph(source_class_graph=source_class_graph, target_class_graph=target_class_graph, doc_sim=doc_sim)
                class_level_graph_generator.to_database(db=db, project_name=project_name, group="auto")







if __name__ == '__main__':
    gen_auto_graph("jgrapht")