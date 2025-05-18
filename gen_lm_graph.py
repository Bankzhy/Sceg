import csv
from pathlib import Path

import pymysql

from gen_merged import project_path_dict
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
    "jgrapht": Path(r"/Users/zhang/Documents/work/jgrapht/jgrapht-core")
}
project_auto_dict = {
    "jgrapht": Path(r"/Users/zhang/Documents/work/jgrapht_auto")
}

def gen_original_graph(project_name):
    project_path = project_path_dict[project_name]
    ast = KASTParse(project_path, "java")
    ast.setup()
    sr_project = ast.do_parse()
    for program in sr_project.program_list:
        for sr_class in program.class_list:
            for sr_method in sr_class.method_list:
                pdg_generator = PDGGenerator(
                    sr_class=sr_class,
                    sr_method=sr_method
                )
                pdg_generator.create_graph()
                pdg_generator.to_database(db=db, project_name=project_name, group="origin")

def gen_auto_graph(project_name):
    auto_index_path = project_auto_dict[project_name] / "index.csv"
    with open(auto_index_path, mode='r') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index > 0:
                path = project_auto_dict[project_name] / row[0]
                ast = KASTParse(path, "java")
                ast.setup()
                file = open(path, encoding='utf-8')
                file_content = file.read()
                sr_project = ast.do_parse_content(file_content)
                for program in sr_project.program_list:
                    for sr_class in program.class_list:
                        for sr_method in sr_class.method_list:
                            if sr_method.method_name == row[2] and len(sr_method.param_list) == row[3]:
                                sr_method.method_name = row[4]+"And"+sr_method.method_name
                                pdg_generator = PDGGenerator(
                                    sr_class=sr_class,
                                    sr_method=sr_method
                                )
                                pdg_generator.create_graph()
                                pdg_generator.to_database(db=db, project_name=project_name, group="auto")


if __name__ == '__main__':
    gen_original_graph("jgrapht")