import json

import pymysql

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


def fetch_extract_methods(extract_methods):
    result = []
    eml = extract_methods.split(";")
    for m in eml:
        if "," in m:
            result.extend(m.split(","))
        else:
            result.append(m)
    return result


def eval_refact():
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    methods_dict = {}
    extract_methods_dict = {}
    predict_extract_methods_dict = {}

    cursor = db.cursor()
    print("loading test 1...")
    cursor.execute("SELECT * FROM lm_master where `project` in ('jsprit', 'oh', 'openrefine') and label=1")
    for row in cursor.fetchall():
        project = row[1]
        class_name = row[2]
        label = row[9]
        extract_methods = row[4]
        lc_graph = row[7]

        key = project + "_" + class_name

        methods = []
        for node in lc_graph['nodes']:
            if node['type'] == 'method':
               methods.append(node['name'])
        methods_dict[key] = methods


        extract_methods = fetch_extract_methods(extract_methods)


        extract_methods = list(set(extract_methods))
        extract_methods_dict[key] = extract_methods

    with open('index.csv') as f:
        lines = f.readlines()
        for line in lines:
            ll = line.split(',')
            project = ll[0]
            class_name = ll[1]
            predict_extract_methods = ll[2]
            predict_extract_methods = fetch_extract_methods(predict_extract_methods)
            predict_extract_methods = list(set(predict_extract_methods))

            key = project + "_" + class_name

            if key in methods_dict.keys():
                for method in methods_dict[key]:
                    if method in predict_extract_methods:
                        if method in extract_methods_dict[key]:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if method in extract_methods_dict[key]:
                            FN += 1
                        else:
                            TN += 1

    eval(TP, TN, FP, FN)



def eval_detect():
    TP = 0
    FN = 0
    FP = 0
    TN = 0


    labels = []

    with open('index.csv') as f:
        lines = f.readlines()
        for line in lines:
            ll = line.split(',')
            project = ll[0]
            class_name = ll[1]

            key = project + "_" + class_name
            labels.append(key)

    cursor = db.cursor()
    print("loading test 1...")
    cursor.execute("SELECT * FROM lm_master where `project` in ('jsprit', 'oh', 'openrefine')")
    for row in cursor.fetchall():
        project = row[1]
        class_name = row[2]
        label = row[9]
        key = project + "_" + class_name
        if label == 1:
            if key in labels:
                TP += 1
            else:
                FN += 1
        else:
            if key in labels:
                FP += 1
            else:
                TN += 1
    eval(TP, TN, FP, FN)


if __name__ == '__main__':
    eval_detect()