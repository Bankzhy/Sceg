import pymysql

db = pymysql.connect(
    host="47.113.220.80",
    user="root",
    password="Apple3328823%",
    database="sce",
    charset="utf8mb4",  # Use utf8mb4 for full Unicode support
    connect_timeout=50
)


def eval_refact():
    all = 0
    right = 0

    target_class_dict = {}
    loc_dict = {}

    cursor = db.cursor()
    print("loading test 1...")
    cursor.execute("SELECT * FROM fe_master where `project` in ('jsprit', 'oh', 'openrefine') and label=1")
    for row in cursor.fetchall():
        fe_project = row[1]
        fe_class_name = row[2]
        fe_method_name = row[3]
        fe_target_class_name = row[5]

        fe_key = fe_project + '_' + fe_class_name + '_' + fe_method_name
        target_class_dict[fe_key] = fe_target_class_name

    with open('index.csv') as f:
        lines = f.readlines()
        for line in lines:
            ll = line.split(',')
            project = ll[0]
            class_name = ll[1]
            method_name = ll[2]
            target_class_name = ll[3]


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
    cursor.execute("SELECT * FROM fe_master where `project` in ('jsprit', 'oh', 'openrefine')")
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

if __name__ == '__main__':
    eval_detect()