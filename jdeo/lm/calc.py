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

def eval_refact():
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    label_dict = {}
    loc_dict = {}

    cursor = db.cursor()
    print("loading test 1...")
    cursor.execute("SELECT * FROM lm_master where `project` in ('jsprit', 'oh', 'openrefine') and label=1")
    for row in cursor.fetchall():
        lm_project = row[1]
        lm_class_name = row[3]
        lm_method_name = row[4]
        lm_label = row[10]
        lm_extract_lines = row[5]
        lm_extract_lines = fetch_extract_line_numbers(lm_extract_lines)

        lm_graph = row[8]
        lm_graph = json.loads(lm_graph)
        nodes = lm_graph["nodes"]
        loc = nodes[0]["metrics"]["loc"]

        lm_key = lm_project + '_' + lm_class_name + '_' + lm_method_name
        label_dict[lm_key] = lm_extract_lines
        loc_dict[lm_key] = loc



    with open('index.csv') as f:
        lines = f.readlines()
        for line in lines:
            ll = line.split(',')
            project = ll[0]
            class_name = ll[1]
            method_name = ll[2]
            extract_lines = ll[3]
            extract_lines = fetch_extract_line_numbers(extract_lines)



            key = project + "_" + class_name + "_" + method_name

            if key in label_dict.keys():
                for i in range(1, len(label_dict[key])+1):
                    if i in extract_lines:
                        if i in label_dict[key]:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if i in label_dict[key]:
                            FN += 1
                        else:
                            TN += 1

    eval(TP, TN, FP, FN)


def eval_dect():
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
            method_name = ll[2]

            key = project + "_" + class_name + "_" + method_name
            labels.append(key)

    cursor = db.cursor()
    print("loading test 1...")
    cursor.execute("SELECT * FROM lm_master where `project` in ('jsprit', 'oh', 'openrefine')")
    for row in cursor.fetchall():
        lm_project = row[1]
        lm_class_name = row[3]
        lm_method_name = row[4]
        lm_label = row[10]

        lm_key = lm_project + '_' + lm_class_name + '_' + lm_method_name

        if lm_label == 1:
            if lm_key in labels:
                TP += 1
            else:
                FN += 1
        else:
            if lm_key in labels:
                FP += 1
            else:
                TN += 1
    eval(TP, TN, FP, FN)



def check_mark():
    check = []
    error = []
    with open('index.csv') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index == 0:
                continue
            ll = line.split(',')
            project = ll[0]
            if project == 'jsprt':
                project = 'jsprit'
            class_name = ll[1]
            method_name = ll[2]
            cursor = db.cursor()
            query = "SELECT * FROM lm_master where `project`=%s and `class_name`=%s and `method_name`=%s;"
            cursor.execute(query, (project, class_name, method_name))
            result=cursor.fetchone()
            if result is None:
                error.append(line)
            else:
                lm_id = result[0]
                label = result[10]

                if label == 9:
                    check.append(lm_id)
    print(len(check))
    print(check)
    print("error:",len(error))
    print(error)






if __name__ == '__main__':
    check_mark()