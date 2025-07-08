import os
from pathlib import Path
import pymysql

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

    pos_count = 0
    with open(dataset_path / "train_1.txt", "w", encoding="utf-8") as f:
        cursor = db.cursor()
        print("loading training 1...")
        cursor.execute("SELECT * FROM lm_master where `label`=1 and split='eval'")
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
            pos_count += 1

    with open(dataset_path / "train_0.txt", "w", encoding="utf-8") as f:
        print("loading remote...")
        cursor.execute("SELECT * FROM lm_master where `label`=0 and split='eval' limit " + str(pos_count) )
        for row in cursor.fetchall():
            lm_id = row[0]
            lm_graph = row[8]
            f.write(lm_graph + "\n")
        f.close()


def run():
   build_training_dataset()
   # build_eval_dataset()


if __name__ == '__main__':
    run()