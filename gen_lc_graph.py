import csv
OVER_SIZE_LIMIT = 200_000_000

csv.field_size_limit(OVER_SIZE_LIMIT)

# 以下、書きたい処理
import json
import os
from pathlib import Path

import pymysql

from common import project_auto_dict
from gen_merged import project_path_dict
from graph.class_level_graph_generator import ClassLevelGraphGenerator
from graph.doc_sim import DocSim
from graph.matrix_construction import MatrixConstruction
from graph.pdg_generator import PDGGenerator
from sitter.kast2core import KASTParse

db = pymysql.connect(
    host="47.113.220.80",
    user="root",
    password="Apple3328823%",
    database="sce",
    charset="utf8mb4",  # Use utf8mb4 for full Unicode support
    connect_timeout=5000
)

def gen_original_graph(project_name):
    project_path = project_path_dict[project_name]
    ast = KASTParse(project_path, "java")
    ast.setup()
    sr_project = ast.do_parse()
    class_list = []
    for program in sr_project.program_list:
        for sr_class in program.class_list:
            class_list.append(sr_class)

    for program in sr_project.program_list:
        for sr_class in program.class_list:
            if sr_class.class_name == "Flowable" or sr_class.class_name == "Observable":
                continue
            class_level_graph_generator=ClassLevelGraphGenerator(sr_class=sr_class, class_list=class_list)
            print(sr_class.class_name)
            class_level_graph_generator.create_graph()
            class_level_graph_generator.to_database(db=db, project_name=project_name, group="original")


def find_class_graph_from_database(name, project):
    cursor = db.cursor()
    query = "select `graph` from lc_master where `class_name` ='" + str(name) + "' and project = '" + str(project) + "';"
    cursor.execute(query)
    data = cursor.fetchall()
    print(data)
    graph = data[0][0]
    graph = json.loads(graph)

    return graph

def gen_auto_graph(project_name):
    path = project_auto_dict[project_name] / "lc/"
    doc_sim = DocSim()
    csv_rows = []

    with open(path / "index.csv", mode='r') as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            file_path = path / row[0]
            ast = KASTParse(file_path, "java")
            ast.setup()
            file = open(file_path, encoding='utf-8')
            file_content = file.read()
            sr_project = ast.do_parse_content(file_content)
            for program in sr_project.program_list:
                for sr_class in program.class_list:
                    try:
                        merged_class_name = sr_class.class_name
                        merged_class_name_l = merged_class_name.split("_")
                        target_class_name = merged_class_name_l[2]
                        source_class_name = merged_class_name_l[3]

                        if source_class_name == "Flowable" or source_class_name == "Observable":
                            continue
                        if target_class_name == "Flowable" or target_class_name == "Observable":
                            continue

                        source_class_graph = find_class_graph_from_database(source_class_name, project_name)
                        target_class_graph = find_class_graph_from_database(target_class_name, project_name)

                        class_level_graph_generator = ClassLevelGraphGenerator(sr_class=sr_class, class_list=[])
                        class_level_graph_generator.create_merge_graph(source_class_graph=source_class_graph, target_class_graph=target_class_graph, doc_sim=doc_sim)
                        # print(row[4])
                        # class_level_graph_generator.to_database(db=db, project_name=project_name, group="auto", extract_methods=row[4])

                        row = class_level_graph_generator.to_list(db=db, project_name=project_name, group="auto", extract_methods=row[4])
                        csv_rows.append(row)
                    except Exception as e:
                        print(e)
                        continue

    file_order = ["project", "class_name", "content", "extract_methods", "group", "split", "graph",
                  "path", "label", "reviewer_id"]
    with open((project_name+"_index.csv"), "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, file_order)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(dict(zip(file_order, row)))





def from_csv():
    cursor = db.cursor()

    with open('grootindex.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            print(row)
            try:
                query = (
                    r"replace into lc_master (project, class_name, content, extract_methods, `group`, split, graph, `path`, label, reviewer_id) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
                values = (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
                cursor.execute(query, values)
                db.commit()
            except Exception as e:
                print(e)
                continue


def mark_pos_nodes():
    cursor = db.cursor()
    # target = [19039, 19040, 19041, 19042, 19043, 19044, 19045, 19046, 19047, 19048, 19049, 19050, 19051, 19052, 19053, 19054, 19055, 19056, 19057, 19058, 19059, 19060, 19061, 19062, 19063, 19064, 19065, 19066, 19067, 19068, 19069, 19070, 19071, 19072, 19073, 19074, 19075, 19076, 19077, 19078, 19079, 19080, 19081, 19082, 19083, 19084, 19085, 19086, 19087, 19088, 19089, 19090, 19091, 19092, 19093, 19094, 19095, 19096, 19097, 19098, 19099, 19100, 19101, 19102, 19103, 19104, 19105, 19106, 19107, 19108, 19109, 19110, 19111, 19112, 19113, 19114, 19115, 19116, 19117, 19118, 19119, 19120, 19121, 19122, 19123, 19124, 19125, 19126, 19127, 19128, 19129, 19130, 19131, 19132, 19133, 19134, 19135, 19136, 19137, 19138, 19139, 19140, 19141, 19142, 19143, 19144, 19145, 19146, 19147, 19148, 19149, 19150, 19151, 19152, 19153, 19154, 19155, 19156, 19157, 19158, 19159, 19160, 19161, 19162, 19163, 19164, 19165, 19166, 19167, 19168, 19169, 19170, 19171, 19172, 19173, 19174, 19175, 19176, 19177, 19178, 19179, 19180, 19972, 19975, 19978, 19980, 19981, 19982, 19983, 19990, 19991, 19993, 19994, 19995, 20007, 20012, 20015, 20016, 20024, 20053, 20058, 20061, 20069, 20071, 20073, 20075, 20076, 20077, 20079, 20080, 20082, 20084, 20086, 20093, 20095, 20134, 20153, 20173, 20178, 20183, 20247, 20250, 20281, 20288, 20301, 20304, 20345, 20355, 20375, 20378, 20387, 20423, 20430, 20462, 20527, 20550, 20574, 20603, 20609, 20617, 20618, 20619, 20620, 20621, 20623, 20625, 20631, 20636, 20641, 20643, 20644, 20648, 20649, 20653, 20654, 20655, 20656, 20657, 20658, 20659, 20660, 20661, 20662, 20663, 20664, 20665, 20666, 20668, 20669, 20671, 20673, 20674, 20676, 20679, 20681, 20711, 20894, 20931, 20933, 20947, 20948, 20958, 20969, 20974, 20977, 20978, 20979, 20980, 20982, 20984, 20985, 20986, 20987, 20988, 20989, 20993, 20994, 20996, 21006, 21025, 21028, 21042, 21045, 21046, 21047, 21048, 21049, 21050, 21051, 21052, 21053, 21054, 21056, 21062, 21063, 21077, 21093, 21094, 21095, 21108, 21109, 21110, 21111, 21112, 21113, 21114, 21115, 21116, 21117, 21118, 21119, 21120, 21121, 21122, 21123, 21124, 21125, 21126, 21127, 21128, 21129, 21130, 21131, 21132, 21133, 21134, 21135, 21136, 21137, 21138, 21139, 21140, 21141, 21142, 21143, 21144, 21145, 21146, 21147, 21148, 21149, 21150, 21151, 21152, 21153, 21154, 21155, 21156, 21157, 21158, 21161, 21162, 21169, 21170, 21175, 21176, 21177, 21178, 21179, 21180, 21181, 21182, 21183, 21184, 21192, 21196, 21197, 21209, 21214, 21216, 21223, 21224, 21232, 21233, 21234, 21235, 21236, 21237, 21238, 21245, 21246, 21247, 21248, 21251, 21252, 21255, 21256, 21260, 21261, 21267, 21268, 21271, 21288, 21303, 21307, 21308, 21310, 21315, 21316, 21317, 21318, 21319, 21320, 21321, 21322, 21327, 21328, 21329, 21330, 21331, 21332, 21333, 21334, 21335, 21341, 21345, 21347, 21348, 21349, 21350, 21351, 21352, 21354, 21355, 21356, 21357, 21358, 21359, 21360, 21361, 21362, 21363, 21364, 21365, 21366, 21367, 21368, 21369, 21375, 21376, 21377, 21378, 21379, 21380, 21381, 21382, 21383, 21384, 21385, 21386, 21387, 21388, 21389, 21390, 21391, 21392, 21393, 21394, 21395, 21396, 21397, 21398, 21399, 21400, 21401, 21402, 21403, 21404, 21405, 21406, 21407, 21408, 21409, 21410, 21411, 21412, 21413, 21415, 21416, 21417, 21418, 21419, 21420, 21422, 21430, 21442, 21443, 21484, 21485, 21486, 21487, 21488, 21489, 21491, 21492, 21494, 21495, 21497, 21501, 21502, 21503, 21505, 21506, 21507, 21508, 21509, 21510, 21511, 21516, 21517, 21545, 21546, 21553, 21555, 21561, 21562, 21563, 21564, 21565, 21567, 21568, 21570, 21573, 21574, 21577, 21579, 21583, 21584, 21585, 21586, 21587, 21588, 21589, 21592, 21594, 21596, 21597, 21598, 21599, 21608, 21609, 21610, 21611, 21612, 21646, 21648, 21650, 21651, 21652, 21654, 21655, 21657, 21658, 21659, 21660, 21661, 21663, 21664, 21671, 21688, 21689, 21697, 21698, 21699, 21700, 21701, 21702, 21715, 21716, 21717, 21718, 21719, 21720, 21721, 21722, 21723, 21731, 21733, 21734, 21738, 21739, 21741, 21742, 21743, 21744, 21745, 21746, 21747, 21748, 21749, 21750, 21753, 21754, 21755, 21756, 21757, 21763, 21764, 21765, 21766, 21767, 21768, 21769, 21770, 21771, 21774, 21777, 21779, 21780, 21781, 21782, 21783, 21784, 21785, 21786, 21787, 21789, 21790, 21794, 21800, 21801, 21814, 21820, 21821, 21822, 21823, 21824, 21825, 21826, 21827, 21828, 21829, 21830, 21831, 21832, 21833, 21834, 21835, 21836, 21837, 21838, 21864, 21865, 21866, 21867, 21868, 21869, 21870, 21871, 21872, 21873, 21879, 21880, 21881, 21882, 21883, 21884, 21885, 21889, 21891, 21892, 21893, 21895, 21898, 21901, 21902, 21903, 21904, 21906, 21907, 21916, 21917, 21925, 21931, 21932, 21933, 21934, 21935, 21936, 21937, 21938, 21939, 21940, 21941, 21942, 21943, 21944, 21945, 21946, 21947, 21948, 21949, 21950, 21951, 21952, 21953, 21954, 21955, 21956, 21957, 21958, 21959, 21960, 21961, 21962, 21963, 21964, 21965, 21966, 21967, 21974, 21990, 21991, 21992, 22043, 22044, 22052, 22053, 22058, 22059, 22060, 22061, 22062, 22063, 22064, 22065, 22066, 22067, 22068, 22069, 22070, 22071, 22072, 22073, 22077, 22078, 22082, 22083, 22085, 22088, 22092, 22093, 22094, 22095, 22096, 22097, 22098, 22099, 22100, 22101, 22102, 22113, 22114, 22115, 22128, 22129, 22131, 22142, 22145, 22154, 22155, 22165, 22171, 22172, 22181, 22184, 22186, 22190, 22191, 22192, 22193, 22196, 22197, 22198, 22200, 22201, 22202, 22203, 22204, 22205, 22206, 22207, 22208, 22209, 22210, 22211, 22216, 22217, 22220, 22222, 22225, 22226, 22227, 22229, 22234, 22236, 22247, 22248, 22249, 22267, 22268, 22269, 22271, 22273, 22274, 22275, 22276, 22277, 22279, 22280, 22281, 22282, 22283, 22284, 22285, 22286, 22288, 22290, 22291, 22292, 22293, 22294, 22295, 22298, 22299, 22300, 22303, 22306, 22311, 22312, 22331, 22345, 22351, 22352, 22353, 22354, 22355, 22356, 22357, 22358, 22359, 22360, 22361, 22362, 22363, 22364, 22365, 22366, 22367, 22368, 22369, 22370, 22372, 22373, 22385, 22386, 22387, 22388, 22389, 22390, 22397, 22400, 22407, 22408, 22409, 22410, 22411, 22412, 22413, 22414, 22415, 22416, 22424, 22427, 22428, 22429, 22430, 22431, 22432, 22433, 22434, 22444, 22453, 22456, 22457, 22458, 22459, 22460, 22461, 22462, 22463, 22464, 22465, 22469, 22470, 22472, 22473, 22474, 22475, 22476, 22478, 22479, 22480, 22481, 22482, 22483, 22484, 22485, 22486, 22487, 22488, 22489, 22490, 22491, 22492, 22493, 22494, 22495, 22496, 22497, 22498, 22499, 22500, 22501, 22502, 22503, 22504, 22505, 22506, 22507, 22509, 22510, 22511, 22512, 22514, 22515, 22516, 22517, 22518, 22519, 22520, 22522, 22524, 22527, 22528, 22529, 22530, 22531, 22534, 22535, 22536, 22537, 22538, 22539, 22542, 22543, 22544, 22546, 22550, 22552, 22553, 22554, 22556, 22557, 22559, 22560, 22561, 22562, 22564, 22565, 22566, 22567, 22568, 22569, 22570, 22571, 22572, 22573, 22574, 22575, 22576, 22577, 22578, 22579, 22580, 22581, 22587, 22588, 22590, 22601, 22619, 22623, 22624, 22629, 22631, 22660, 22661, 22662, 22663, 22664, 22665, 22666, 22667, 22668, 22682, 22689, 22691, 22703, 22704, 22705, 22718, 22719, 22720, 22722, 22723, 22724, 22736, 22743, 22760, 22761, 22762, 22763, 22764, 22765, 22766, 22767]
    query = "SELECT * FROM lc_master where `label`=1 and extract_methods!='' and lc_id in (21962, 21963, 22312, 22407, 22601, 22682, 22689, 22691, 22703)"
    # target = [str(x) for x in target]
    cursor.execute(query)
    failed = []
    rows = cursor.fetchall()
    print("total: ", len(rows))
    for row in rows:
        # try:
            lc_id = row[0]
            lc_name = row[2]
            lc_graph = row[7]
            lc_graph = json.loads(lc_graph)
            extract_methods = row[4]
            extract_methods = extract_methods.split(",")
            group = row[5]

            if lc_name.startswith("ihc") or lc_name.startswith("ilc"):
                project_name = row[1]
                merged_class_name = row[2]
                merged_class_name_l = merged_class_name.split("_")
                target_class_name = merged_class_name_l[2]
                source_class_name = merged_class_name_l[3]
                source_class_graph = find_class_graph_from_database(source_class_name, project_name)
                target_class_graph = find_class_graph_from_database(target_class_name, project_name)

                merge_method_nodes = []
                for node in source_class_graph["nodes"]:
                    if node["type"] == "class":
                        source_class_node = node
                    else:
                        merge_method_nodes.append(node)
                for node in target_class_graph["nodes"]:
                    if node["type"] == "class":
                        target_class_node = node
                    else:
                        merge_method_nodes.append(node)

                nodes = lc_graph["nodes"]
                for index, node in enumerate(nodes):
                    if index == 0:
                        continue

                    if merge_method_nodes[index-1] in extract_methods:
                        node["name"] = merge_method_nodes[index-1]["name"]
                        node["is_extract"] = 1
                    else:
                        node["name"] = merge_method_nodes[index - 1]["name"]
                        node["is_extract"] = 0
                lc_graph_str = json.dumps(lc_graph)
                query = (r"update lc_master set graph=%s where lc_id=%s;")
                values = (lc_graph_str, lc_id)
                print("update: ", str(lc_id))
                cursor.execute(query, values)

            else:
                nodes = lc_graph["nodes"]
                for index, node in enumerate(nodes):
                    if index == 0:
                        continue

                    if node["name"] in extract_methods:
                        node["is_extract"] = 1
                    else:
                        node["is_extract"] = 0
                lc_graph_str = json.dumps(lc_graph)
                query = (r"update lc_master set graph=%s where lc_id=%s;")
                values = (lc_graph_str, lc_id)
                print("update: ", str(lc_id))
                cursor.execute(query, values)
        # except Exception as e:
        #     print(e)
        #     failed.append(lc_id)

    db.commit()
    print(failed)
    print(nodes)



def fix_auto_graph():
    cursor = db.cursor()
    cursor.execute("select * from lc_master where graph not like '%class_loc%';")
    for row in cursor.fetchall():
        lc_id = row[0]
        project_name = row[1]
        lc_class_name = row[2]
        lc_graph = row[7]
        lc_graph = json.loads(lc_graph)

        merged_class_name = lc_class_name
        merged_class_name_l = merged_class_name.split("_")
        target_class_name = merged_class_name_l[2]
        source_class_name = merged_class_name_l[3]

        if source_class_name == "Flowable" or source_class_name == "Observable":
            continue
        if target_class_name == "Flowable" or target_class_name == "Observable":
            continue

        source_class_graph = find_class_graph_from_database(source_class_name, project_name)
        target_class_graph = find_class_graph_from_database(target_class_name, project_name)

        # merged_fields = source_class_graph['nodes'][0]["fields"] + "," + target_class_graph["nodes"][0]["fields"]
        # lc_graph["nodes"][0]["fields"] = merged_fields
        source_class_loc = source_class_graph["nodes"][0]["metrics"]["class_loc"]
        target_class_loc = target_class_graph["nodes"][0]["metrics"]["class_loc"]
        class_loc = source_class_loc+target_class_loc
        lc_graph["nodes"][0]["metrics"]["class_loc"] = class_loc

        lc_graph_str = json.dumps(lc_graph)

        # print(lc_graph_str)
        print("=====================")

        query = (r"update lc_master set graph=%s where lc_id=%s;")
        values = (lc_graph_str, lc_id)
        print(query)
        cursor.execute(query, values)

    db.commit()





if __name__ == '__main__':
    # for index, key in enumerate(project_path_dict.keys()):
    #     print("=================================")
    #     print(key)
    #     gen_original_graph(key)
    # for index, key in enumerate(project_auto_dict.keys()):
    #     if index == 0:
    #         continue
    #     if index == 1:
    #         continue
    #     if index == 9:
    #         continue
    #     print("=================================")
    #     print(key)
    #     print("=================================")
    #     gen_auto_graph(key)
    # gen_auto_graph("rxJava")
    # from_csv()
    # mark_pos_nodes()
    fix_auto_graph()