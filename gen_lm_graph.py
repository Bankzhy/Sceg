import csv
import os.path
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
    "jgrapht": Path(r"D:\research\code_corpus\jgrapht\jgrapht-core")
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
            for sr_method in sr_class.method_list:
                try:
                    pdg_generator = PDGGenerator(
                        sr_class=sr_class,
                        sr_method=sr_method
                    )
                    pdg_generator.create_graph()
                    pdg_generator.to_database(db=db, project_name=project_name, group="original")
                except Exception as e:
                    print("Error:")
                    print(e)


def find_extract_lines(method_content, extract_content):
    result = ""
    extract_lines = extract_content.split("\n")
    extract_start_line = extract_lines[0].replace(" ", "")
    extract_second_line = extract_lines[1].replace(" ", "")

    method_lines = method_content.split("\n")
    for index, line in enumerate(method_lines):
        if index < (len(method_lines)-1):
            strip_line = line.replace(" ", "")
            next_strip_line = method_lines[index+1].replace(" ", "")

        if strip_line == extract_start_line and next_strip_line == extract_second_line:
            result = str(index+1)+"-"+str(index+1+len(extract_lines))
    return result

def gen_auto_graph(project_name):
    auto_file_path = project_auto_dict[project_name] / "lm"
    if os.path.exists(auto_file_path) is False:
        os.mkdir(auto_file_path)
    auto_index_path = auto_file_path / "index.csv"
    with open(auto_index_path, mode='r') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index > 0:
                path = auto_file_path / (row[1] + ".java")
                ast = KASTParse(path, "java")
                ast.setup()
                file = open(path, encoding='utf-8')
                file_content = file.read()
                sr_project = ast.do_parse_content(file_content)
                for program in sr_project.program_list:
                    for sr_class in program.class_list:
                        for sr_method in sr_class.method_list:
                            if sr_method.method_name == row[2] and str(len(sr_method.param_list)) == row[4]:
                                sr_method.method_name = row[4]+"And"+sr_method.method_name

                                extract_lines = find_extract_lines(sr_method.to_string(), row[7])

                                pdg_generator = PDGGenerator(
                                    sr_class=sr_class,
                                    sr_method=sr_method
                                )
                                pdg_generator.create_graph()
                                pdg_generator.to_database(db=db, project_name=project_name, group="auto", extract_lines=extract_lines)


if __name__ == '__main__':
    # gen_auto_graph("jgrapht")
    gen_original_graph("jgrapht")