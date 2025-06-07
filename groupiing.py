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
MIN_LOC = 15
MAX_LOC = 30
MIN_CLS_LOC = 70
MAX_CLS_LOC = 130
MIN_NOM = 7
MAX_NOM = 10
MIN_NOA = 5
MAX_NOA = 70
MIN_NFDI = 2
MAX_NFDI = 5

def grouping_lm_original():
    group_a_ids = []
    group_m_ids = []

    cursor = db.cursor()
    cursor.execute("SELECT * FROM lm_master where `group`='original' and split='train'")
    for row in cursor.fetchall():
        lm_id = row[0]
        lm_graph = row[8]
        print(lm_id)
        lm_graph = json.loads(lm_graph)

        loc = None
        nodes = lm_graph["nodes"]
        for node in nodes:
            if "type" in node.keys():
                loc = node["metrics"]["loc"]
                break


        if loc <= MIN_LOC:
            group_a_ids.append(str(lm_id))
        elif loc == None:
            continue
        else:
            group_m_ids.append(str(lm_id))

    if len(group_a_ids) > 0:
        placeholders = ','.join(group_a_ids)
        query = r"update lm_master set `group` = %s, label=%s where lm_id in ("+placeholders+");"
        values = ("a", "0")
        print(query)
        cursor.execute(query, values)
        db.commit()

    if len(group_m_ids) > 0:
        placeholders = ','.join(group_m_ids)
        query = r"update lm_master set `group` = %s where lm_id in ("+placeholders+");"
        values = ("m")
        print(query)
        cursor.execute(query, values)
        db.commit()


def grouping_lm_auto():
    group_a_ids = []
    group_m_ids = []

    cursor = db.cursor()
    cursor.execute("SELECT * FROM lm_master where `group`='auto' and split='train'")
    for row in cursor.fetchall():
        lm_id = row[0]
        lm_graph = row[8]
        lm_graph = json.loads(lm_graph)
        loc = None
        nodes = lm_graph["nodes"]
        for node in nodes:
            if "type" in node.keys():
                loc = node["metrics"]["loc"]
                break

        if loc <= MIN_LOC:
            group_a_ids.append(str(lm_id))
        elif loc == None:
            continue
        else:
            group_m_ids.append(str(lm_id))

    if len(group_a_ids) > 0:
        placeholders = ','.join(group_a_ids)
        query = r"update lm_master set `group` = %s, label=%s where lm_id in ("+placeholders+");"
        print(query)
        values = ("a", "1")
        cursor.execute(query, values)
        db.commit()

    if len(group_m_ids) > 0:
        placeholders = ','.join(group_m_ids)
        query = r"update lm_master set `group` = %s where lm_id in ("+placeholders+");"
        print(query)
        values = ("m")
        cursor.execute(query, values)
        db.commit()


def grouping_lc_original():
    group_a_ids = []
    group_m_ids = []

    cursor = db.cursor()
    cursor.execute("SELECT * FROM lc_master where `group`='original'")
    for row in cursor.fetchall():
        lc_id = row[0]
        lc_graph = row[7]
        lc_graph = json.loads(lc_graph)
        loc = None
        nom = None
        noa = None
        nodes = lc_graph["nodes"]
        for node in nodes:
            if node["type"] == "class":
                loc = node["metrics"]["loc"]
                nom = node["metrics"]["nom"]
                noa = node["metrics"]["noa"]

        if loc <= MIN_CLS_LOC and noa <= MIN_NOA and nom <= MIN_NOM:
            group_a_ids.append(lc_id)
        elif loc == None:
            continue
        else:
            group_m_ids.append(lc_id)

    # query = r"update lc_master set `group` = %s, label=%s where id in (%s);"
    # values = ("a", 0, ",".join(group_a_ids))
    # cursor.execute(query, values)
    # db.commit()
    #
    # query = r"update lc_master set `group` = %s where id in (%s);"
    # values = ("m", ",".join(group_m_ids))
    # cursor.execute(query, values)
    # db.commit()

    if len(group_a_ids) > 0:
        placeholders = ','.join(group_a_ids)
        query = r"update lc_master set `group` = %s, label=%s where id in ("+placeholders+");"
        values = ("a", "0")
        print(query)
        cursor.execute(query, values)
        db.commit()

    if len(group_m_ids) > 0:
        placeholders = ','.join(group_m_ids)
        query = r"update lc_master set `group` = %s where id in ("+placeholders+");"
        values = ("m")
        print(query)
        cursor.execute(query, values)
        db.commit()



def grouping_lc_auto():
    group_a_ids = []
    group_m_ids = []

    cursor = db.cursor()
    cursor.execute("SELECT * FROM lc_master where `group`='auto'")
    for row in cursor.fetchall():
        lc_id = row[0]
        lc_graph = row[7]
        lc_graph = json.loads(lc_graph)
        loc = None
        nom = None
        noa = None
        nodes = lc_graph["nodes"]
        for node in nodes:
            if node["type"] == "class":
                loc = node["metrics"]["loc"]
                nom = node["metrics"]["nom"]
                noa = node["metrics"]["noa"]

        if loc > MIN_CLS_LOC and noa > MIN_NOA and nom > MIN_NOM:
            group_a_ids.append(lc_id)
        elif loc == None:
            continue
        else:
            group_m_ids.append(lc_id)

    # query = r"update lc_master set `group` = %s, label=%s where id in (%s);"
    # values = ("a", 1, ",".join(group_a_ids))
    # cursor.execute(query, values)
    # db.commit()
    #
    # query = r"update lc_master set `group` = %s where id in (%s);"
    # values = ("m", ",".join(group_m_ids))
    # cursor.execute(query, values)
    # db.commit()

    if len(group_a_ids) > 0:
        placeholders = ','.join(group_a_ids)
        query = r"update lc_master set `group` = %s, label=%s where id in ("+placeholders+");"
        print(query)
        values = ("a", "1")
        cursor.execute(query, values)
        db.commit()

    if len(group_m_ids) > 0:
        placeholders = ','.join(group_m_ids)
        query = r"update lc_master set `group` = %s where id in ("+placeholders+");"
        print(query)
        values = ("m")
        cursor.execute(query, values)
        db.commit()


def grouping_fe_original():
    group_a_ids = []
    group_m_ids = []

    cursor = db.cursor()
    cursor.execute("SELECT * FROM fe_master where `group`='original'")
    for row in cursor.fetchall():
        fe_id = row[0]
        fe_graph = row[7]
        target_method_name = row[2]
        fe_graph = json.loads(fe_graph)

        nfdi = None
        nodes = fe_graph["nodes"]
        for node in nodes:
            if node["type"] == "method":
                if node["name"] == target_method_name:
                    nfdi = node["metrics"]["nfdi"]

        if nfdi <= MIN_NFDI:
            group_a_ids.append(fe_id)
        elif nfdi == None:
            continue
        else:
            group_m_ids.append(fe_id)

    # query = r"update fe_master set `group` = %s, label=%s where id in (%s);"
    # values = ("a", 0, ",".join(group_a_ids))
    # cursor.execute(query, values)
    # db.commit()
    #
    # query = r"update fe_master set `group` = %s where id in (%s);"
    # values = ("m", ",".join(group_m_ids))
    # cursor.execute(query, values)
    # db.commit()

    if len(group_a_ids) > 0:
        placeholders = ','.join(group_a_ids)
        query = r"update fe_master set `group` = %s, label=%s where id in ("+placeholders+");"
        values = ("a", "0")
        print(query)
        cursor.execute(query, values)
        db.commit()

    if len(group_m_ids) > 0:
        placeholders = ','.join(group_m_ids)
        query = r"update fe_master set `group` = %s where id in ("+placeholders+");"
        values = ("m")
        print(query)
        cursor.execute(query, values)
        db.commit()

def grouping_fe_auto():
    group_a_ids = []
    group_m_ids = []

    cursor = db.cursor()
    cursor.execute("SELECT * FROM fe_master where `group`='auto'")
    for row in cursor.fetchall():
        fe_id = row[0]
        fe_graph = row[7]
        target_method_name = row[2]
        fe_graph = json.loads(fe_graph)

        nfdi = None
        nodes = fe_graph["nodes"]
        for node in nodes:
            if node["type"] == "method":
                if node["name"] == target_method_name:
                    nfdi = node["metrics"]["nfdi"]

        if nfdi > MAX_NFDI:
            group_a_ids.append(fe_id)
        elif nfdi == None:
            continue
        else:
            group_m_ids.append(fe_id)

    # query = r"update fe_master set `group` = %s, label=%s where id in (%s);"
    # values = ("a", 1, ",".join(group_a_ids))
    # cursor.execute(query, values)
    # db.commit()
    #
    # query = r"update fe_master set `group` = %s where id in (%s);"
    # values = ("m", ",".join(group_m_ids))
    # cursor.execute(query, values)
    # db.commit()

    if len(group_a_ids) > 0:
        placeholders = ','.join(group_a_ids)
        query = r"update fe_master set `group` = %s, label=%s where id in ("+placeholders+");"
        print(query)
        values = ("a", "1")
        cursor.execute(query, values)
        db.commit()

    if len(group_m_ids) > 0:
        placeholders = ','.join(group_m_ids)
        query = r"update fe_master set `group` = %s where id in ("+placeholders+");"
        print(query)
        values = ("m")
        cursor.execute(query, values)
        db.commit()


if __name__ == '__main__':
    grouping_lm_auto()