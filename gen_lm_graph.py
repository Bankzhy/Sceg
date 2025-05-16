from pathlib import Path

import pymysql

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
cursor = db.cursor()

def gen_original_graph():
    project_path = Path(r"D:\research\code_corpus\jgrapht\jgrapht-core\src\main\java")
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
                pdg_generator.to_database(cursor=cursor)

if __name__ == '__main__':
    gen_original_graph()