import csv
import os.path
from pathlib import Path

import pymysql

from common import project_auto_dict, project_path_dict
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

# project_path_dict = {
#     "jgrapht": Path(r"D:\research\code_corpus\jgrapht\jgrapht-core"),
#     "libgdx": Path(r"D:\research\code_corpus\libgdx\gdx"),
#     "freeplane": Path(r"D:\research\code_corpus\freeplane\freeplane\src\main"),
#     "jsprit": Path(r"F:\research\dataset\op\jsprit\jsprit-core\src\main"),
#     "oh": Path(r"F:\research\dataset\op\openhospital-core\src\main\java"),
#     "openrefine": Path(r"D:\research\eval_projects\OpenRefine\main\src"),
#
#     "jedit": Path(r"D:\research\code_corpus\jEdit\org\jedit"),
#     "rxJava": Path(r"D:\research\code_corpus\RxJava\src\main\java"),
#     "junit4": Path(r"D:\research\code_corpus\junit4\src\main"),
#     "mybatis3": Path(r"D:\research\code_corpus\mybatis-3\src\main"),
#     "netty": Path(r"D:\research\code_corpus\netty\codec-base\src\main"),
#     "gephi": Path(r"D:\research\code_corpus\gephi\modules"),
#     "plantuml": Path(r"D:\research\code_corpus\plantuml\src\main"),
#     "groot": Path(r"D:\research\code_corpus\groot\src\main"),
#     "musicBot": Path(r"D:\research\code_corpus\MusicBot\src\main"),
#     "traccar": Path(r"D:\research\code_corpus\traccar\src\main")
# }
# project_auto_dict = {
#     "jedit": Path(r"D:\research\code_corpus\jEdit_auto"),
#     "rxJava": Path(r"D:\research\code_corpus\RxJava_auto"),
#     "junit4": Path(r"D:\research\code_corpus\junit4_auto"),
#     "mybatis3": Path(r"D:\research\code_corpus\mybatis_auto"),
#     "netty": Path(r"D:\research\code_corpus\netty_auto"),
#     "gephi": Path(r"D:\research\code_corpus\gephi_auto"),
#     "plantuml": Path(r"D:\research\code_corpus\plantuml_auto"),
#     "groot": Path(r"D:\research\code_corpus\groot_auto"),
#     "musicBot": Path(r"D:\research\code_corpus\MusicBot_auto"),
#     "traccar": Path(r"D:\research\code_corpus\traccar_auto")
# }

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
                    pdg_generator.class_list = program.class_list
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
    with open(auto_index_path, mode='r', encoding="utf-8") as file:
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
                                pdg_generator.to_database(db=db, project_name=project_name+"_auto", group="auto", extract_lines=extract_lines)


if __name__ == '__main__':
    for key in project_auto_dict.keys():
        print(key)
        gen_auto_graph(key)
    # gen_original_graph("jgrapht")