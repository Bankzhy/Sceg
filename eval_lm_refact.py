import csv
import json

import pymysql
import torch

from dataset.lm.lmr_dataset import LMRDataset

db = pymysql.connect(
    host="47.113.220.80",
    user="root",
    password="Apple3328823%",
    database="sce",
    charset="utf8mb4",  # Use utf8mb4 for full Unicode support
    connect_timeout=50
)

def eval(tp, tn, fp, fn):
    print("tp : ", tp)
    print("tn : ", tn)
    print("fp : ", fp)
    print("fn : ", fn)
    P = tp * 1.0 / (tp + fp)
    R = tp * 1.0 / (tp + fn)
    print("Precision : ", P)
    print("Recall : ", R)
    print("F1 : ", 2 * P * R / (P + R))
    if tp == 0 or tn == 0 or fp == 0 or fn == 0:
        return 1

    a = tp + fp
    b = tp + fn
    c = tn + fp
    d = tn + fn
    print("MCC : ", (tp * tn - fp * fn) / ((a * b * c * d) ** 0.5))

    return 2 * P * R / (P + R)

def fetch_extract_line_numbers(extract_lines):
    result = []
    exl = extract_lines.split(";")
    for ex in exl:
        if "-" in ex:
            el = ex.split("-")
            new_el = list(range(int(int(el[0])), int(el[1])))
            result.extend(new_el)
        else:
            if ex != "":
                result.append(int(ex))
    return result


def model_preds(model, graph):
    st_feats = graph.nodes['statement'].data['feat']
    st_labels = graph.nodes['statement'].data['label']
    node_features = {'statement': st_feats}
    # graph = graph.to(device)
    # st_labels = st_labels.to(device)
    prediction = model(graph, node_features)['statement']
    prediction = prediction.argmax(1).tolist()
    return prediction

def eval_model_refact():
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    jdeo_path = r"jdeo/lm/index.csv"
    model_path = r"output/model/lmr-model-gcn.pkl"
    cursor = db.cursor()
    model = torch.load(model_path)
    model.to("cpu")
    lm_dataset = LMRDataset(jdeo_path)

    with open(jdeo_path, mode='r') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            try:
                project = row[0]
                class_name = row[1]
                method_name = row[2]

                query = (r"SELECT * FROM lm_master where `project` = %s and class_name = %s and method_name = %s")
                cursor.execute(query, (project, class_name, method_name))
                row = cursor.fetchone()
                if row is not None:
                    lm_label = row[10]
                    if lm_label == 1:
                        lm_graph = row[8]
                        lm_extract_lines = row[5]
                        lm_extract_lines = fetch_extract_line_numbers(lm_extract_lines)
                        lm_graph = json.loads(lm_graph)
                        lm_graph = lm_dataset.get_graph(lm_graph)
                        prediction = model_preds(model, lm_graph)

                        for i, p in enumerate(prediction):
                            line_number = i+1
                            if p == 1:
                                if line_number in lm_extract_lines:
                                    TP += 1
                                else:
                                    FP += 1
                            else:
                                if line_number in lm_extract_lines:
                                    FN += 1
                                else:
                                    TN += 1


                        print(prediction)
            except Exception as e:
                print(e)
                continue

    eval(TP, TN, FP, FN)

if __name__ == '__main__':
    eval_model_refact()