import pymysql
import csv

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


def eval_refact():
    all = 0
    right = 0

    target_class_dict = {}
    loc_dict = {}

    cursor = db.cursor()
    print("loading test 1...")
    cursor.execute("SELECT * FROM fe_master where `project` in ('jsprit', 'oh', 'openrefine', 'jgrapht', 'freeplane') and label=1")
    for row in cursor.fetchall():
        fe_project = row[1]
        fe_class_name = row[2]
        fe_method_name = row[3]
        fe_target_class_name = row[5]

        fe_key = fe_project + '_' + fe_class_name + '_' + fe_method_name
        target_class_dict[fe_key] = fe_target_class_name

    with open('index.csv', mode='r') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index == 0:
                continue

            project = row[0]
            class_name = row[1]
            method_name = row[2]
            target_class_name = row[3]


            key = project + "_" + class_name + "_" + method_name

            if key in target_class_dict.keys():
                all += 1
                if target_class_name == target_class_dict[key]:
                    right += 1


    cover = right / all
    print(cover)


def eval_detect():
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    labels = []

    with open('index.csv', mode='r') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index == 0:
                continue

            project = row[0]
            class_name = row[1]
            method_name = row[2]

            key = project + "_" + class_name + "_" + method_name
            labels.append(key)

    cursor = db.cursor()
    print("loading test 1...")
    cursor.execute("SELECT * FROM fe_master where `project` in ('jsprit', 'oh', 'openrefine', 'jgrapht', 'freeplane')")
    for row in cursor.fetchall():
        fe_project = row[1]
        fe_class_name = row[2]
        fe_method_name = row[3]
        fe_label = row[10]

        lm_key = fe_project + '_' + fe_class_name + '_' + fe_method_name

        if fe_label == 1:
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
    with open('index.csv', mode='r') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index == 0:
                continue

            project = row[0]
            if project == 'libgdx':
                continue

            class_name = row[1]
            method_name = row[2]
            cursor = db.cursor()
            query = "SELECT * FROM fe_master where `project`=%s and `class_name`=%s and `method_name`=%s;"
            cursor.execute(query, (project, class_name, method_name))
            result=cursor.fetchone()
            if result is None:
                error.append(row)
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
    eval_refact()