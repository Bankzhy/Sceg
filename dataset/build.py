from pathlib import Path
import pymysql
import os

dataset_path = Path(r"dataset/lm")
db = pymysql.connect(
    host="47.113.220.80",
    user="root",
    password="Apple3328823%",
    database="sce",
    charset="utf8mb4",  # Use utf8mb4 for full Unicode support
    connect_timeout=50
)


def build_training_dataset():

    if os.path.exists(dataset_path / "train_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "train_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading training 1...")
        cursor.execute("SELECT * FROM lm_master where `label`=1 and split='train'")
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
            pos_count += 1

    with open(dataset_path / "train_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM lm_master where `label`=0 and split='train' limit " + str(pos_count) )
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
        f.close()

def build_eval_dataset():

    if os.path.exists(dataset_path / "test_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "test_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading test 1...")
        cursor.execute("SELECT * FROM lm_master where `label`=1 and split='eval'")
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
            pos_count += 1

    with open(dataset_path / "test_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM lm_master where `label`=0 and split='eval' limit " + str(pos_count) )
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
        f.close()

def mark_pos_nodes():
    cursor = db.cursor()
    cursor.execute("SELECT * FROM lm_master where `label`=1 and split='train'")
    for row in cursor.fetchall():
        lm_id = row[0]
        lm_graph = row[8]
        print(lm_id)
        lm_graph = json.loads(lm_graph)
        code = row[2]
        new_nodes = fetch_nodes(code)

        extract_lines = row[5]
        extract_line_numbers = fetch_extract_line_numbers(extract_lines)

        nodes = lm_graph["nodes"]
        for index, node in enumerate(nodes):
            if index == 0:
                continue

            node["start_line"] = 0
            node["end_line"] = 0

            if hasattr(new_nodes[index-1].sr_statement, "start_line"):
                node["start_line"] = new_nodes[index-1].sr_statement.start_line
                node["end_line"] = new_nodes[index - 1].sr_statement.end_line

            statement_line_numbers = list(range(node["start_line"], node["end_line"]+1))
            contains_all = all(elem in extract_line_numbers for elem in statement_line_numbers)
            if contains_all:
                node["is_extract"] = 1
            else:
                node["is_extract"] = 0

        lm_graph_str = json.dumps(lm_graph)
        query = (r"update lm_master set graph=%s where lm_id=%s;")
        values = (lm_graph_str, lm_id)
        cursor.execute(query, values)
        db.commit()

        print(nodes)
