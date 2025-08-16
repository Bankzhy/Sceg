import os

import torch
from pathlib import Path
import pymysql
dataset_path = Path(r"fe")
db = pymysql.connect(
    host="47.113.220.80",
    user="root",
    password="Apple3328823%",
    database="sce",
    charset="utf8mb4",  # Use utf8mb4 for full Unicode support
    connect_timeout=50
)

def build_lm_eval_dataset():

    if os.path.exists(dataset_path / "test_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "test_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading test 1...")
        cursor.execute("SELECT * FROM lm_master where `project` in ('jsprit', 'oh', 'openrefine', 'libgdx') and label=1")
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
            pos_count += 1

    with open(dataset_path / "test_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM lm_master where `project` in ('jsprit', 'oh', 'openrefine', 'libgdx') and label=0;")
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
        f.close()

def build_lm_training_dataset():

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

def build_lc_eval_dataset():

    if os.path.exists(dataset_path / "test_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "test_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading test 1...")
        cursor.execute("SELECT * FROM lc_master where `project` in ('jsprit', 'oh', 'openrefine', 'jgrapht', 'freeplane', 'libgdx') and label=1")
        for row in cursor.fetchall():
            lc_graph = row[7]
            f.write(lc_graph + "\n")
            pos_count += 1

    with open(dataset_path / "test_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM lc_master where `project` in ('jsprit', 'oh', 'openrefine', 'jgrapht', 'freeplane', 'libgdx') and label=0" )
        for row in cursor.fetchall():
            lc_graph = row[7]
            f.write(lc_graph + "\n")
        f.close()

def build_lc_training_dataset():
    if os.path.exists(dataset_path / "train_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "train_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading training 1...")
        cursor.execute("SELECT * FROM lc_master where `label`=1 and split='train'")
        for row in cursor.fetchall():
            lc_graph = row[7]
            f.write(lc_graph + "\n")
            pos_count += 1

    with open(dataset_path / "train_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM lc_master where `label`=0 and split='train' limit " + str(pos_count) )
        for row in cursor.fetchall():
            lc_graph = row[7]
            f.write(lc_graph + "\n")
        f.close()

def build_fe_training_dataset():

    if os.path.exists(dataset_path / "train_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "train_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading training 1...")
        cursor.execute("SELECT * FROM fe_master where `label`=1 and split='train'")
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
            pos_count += 1

    with open(dataset_path / "train_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM fe_master where `label`=0 and split='train' limit " + str(pos_count) )
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
        f.close()

def build_fe_eval_dataset():

    if os.path.exists(dataset_path / "test_1.txt"):
        return

    pos_count = 0
    with open(dataset_path / "test_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading test 1...")
        cursor.execute("SELECT * FROM fe_master where `project` in ('jsprit', 'oh', 'openrefine', 'jgrapht', 'freeplane') and label=1")
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
            pos_count += 1

    with open(dataset_path / "test_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM fe_master where `project` in ('jsprit', 'oh', 'openrefine', 'jgrapht', 'freeplane') and label=0" )
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
        f.close()



if __name__ == '__main__':
    # build_lm_eval_dataset()
    # build_lm_training_dataset()
    # build_lc_eval_dataset()
    # build_lc_training_dataset()
    # build_fe_eval_dataset()
    build_fe_training_dataset()