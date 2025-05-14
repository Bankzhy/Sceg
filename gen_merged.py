import copy
import csv
from pathlib import Path
import re

from reflect.sr_class import SRClass
from reflect.sr_statement import SRStatement
from sitter.ast2core import ASTParse
from sitter.kast2core import KASTParse

class FixObject:

    def __init__(self, target_sr_class, target_method, copy_source_method, program_name, statement):
        self.target_sr_class = target_sr_class
        self.target_method = target_method
        self.copy_source_method = copy_source_method
        self.program_name = program_name
        self.statement = statement

def save_file(text, file_name, path):
    file_name = path / (file_name+".java")
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(text)
        f.close()
    print(file_name, " has been saved...")

def find_sr_cls(sr_cls_n, sr_cls_l):
    for cls in sr_cls_l:
        if str(cls.class_name) == str(sr_cls_n):
            return cls
    return None

def check_common_field(fl1, fl2):
    fl1_n = [o.field_name for o in fl1]
    for f in fl2:
        if f.field_name in fl1_n:
            return True
    return False

def check_common_method(ml1, ml2):
    ml1_n = [o.method_name for o in ml1]
    for m in ml2:
        if m.method_name in ml1_n:
            return True
    return False

def check_ilc_merge_oppturnity(sr_cls, sr_cls_l):
    result_list = []
    failed = 0

    for field in sr_cls.field_list:
        field_cls = find_sr_cls(field.field_type, sr_cls_l)
        if field_cls is not None:
            if field_cls.class_name == sr_cls.class_name:
                failed += 1
                continue
            if check_common_field(field_cls.field_list, sr_cls.field_list) is True \
                    or check_common_method(field_cls.method_list, sr_cls.method_list) is True:
                failed += 1
                continue

            if len(field_cls.extends) > 0:
                field_cls_parent = field_cls.extends[1]
                if len(sr_cls.extends) > 0:
                    if field_cls_parent == sr_cls.extends[1]:
                        new_merge_opp = {
                            "sc": sr_cls,
                            "type": "ilc",
                            "tc": field_cls,
                        }
                        result_list.append(new_merge_opp)
            else:
                new_merge_opp = {
                    "sc": sr_cls,
                    "type": "ilc",
                    "tc": field_cls,
                }
                result_list.append(new_merge_opp)
    return result_list, failed

def check_ihc_merge_oppturnity(sr_cls, sr_cls_l):
    result_list = []
    if len(sr_cls.extends) > 0:
        parent_cls_n = sr_cls.extends[1]
        parent_cls = find_sr_cls(parent_cls_n, sr_cls_l)
        if parent_cls is not None:
            sc_method_l = [o.method_name for o in sr_cls.method_list]
            tc_method_l = [o.method_name for o in parent_cls.method_list]
            extra_method_l = []
            for m in parent_cls.method_list:
                if m.method_name not in sc_method_l:
                    extra_method_l.append(m)
            total_method_loc = 0
            for m in extra_method_l:
                loc = len(m.get_all_statement())
                total_method_loc += loc

            if total_method_loc > 15:
                new_merge_opp = {
                    "sc": sr_cls,
                    "type": "ihc",
                    "tc": parent_cls,
                }
                result_list.append(new_merge_opp)

    return result_list

def gen_lc_opp(opp, index, save_path):
    # opp = obj["opp"]
    # index = obj["index"]
    new_class_name = opp["type"] + "_" + str(index) + "_" + opp["sc"].class_name + "_" + opp["tc"].class_name
    print("current gen cls: " + new_class_name)
    new_merged_class = SRClass(
        class_name=new_class_name,
        id=new_class_name
    )

    if opp['type'] == "ihc":
        new_merged_class.field_list = []
        new_merged_class.method_list = []
        new_merged_class.field_list.extend(opp['sc'].field_list)
        n_f_n_l = [o.field_name for o in new_merged_class.field_list]
        for f in opp['tc'].field_list:
            if f.field_name in n_f_n_l:
                continue
            else:
                new_merged_class.field_list.append(f)

        new_merged_class.method_list.extend(opp['sc'].method_list)
        n_m_n_l = [o.method_name for o in new_merged_class.method_list]
        for m in opp['tc'].method_list:
            if m.method_name in n_m_n_l:
                continue
            else:
                new_merged_class.method_list.append(m)
    elif opp['type'] == "ilc":
        new_merged_class.field_list = []
        new_merged_class.method_list = []
        new_merged_class.field_list.extend(opp['sc'].field_list)
        n_f_n_l = [o.field_name for o in new_merged_class.field_list]
        for f in opp['tc'].field_list:
            if f.field_name in n_f_n_l:
                continue
            else:
                new_merged_class.field_list.append(f)

        new_merged_class.method_list.extend(opp['sc'].method_list)
        n_m_n_l = [o.method_name for o in new_merged_class.method_list]
        for m in opp['tc'].method_list:
            if m.method_name in n_m_n_l:
                continue
            else:
                new_merged_class.method_list.append(m)

    new_merged_class.implement_list = opp['sc'].implement_list
    new_merged_class.extends = opp['sc'].extends
    new_merged_class.modifiers = opp['sc'].modifiers
    new_merged_class.constructor_list = opp['sc'].constructor_list
    new_merged_class.import_list = opp['sc'].import_list
    new_merged_class.package_name = opp['sc'].package_name

    save_file(new_merged_class.to_string(space=0), new_class_name, save_path)

def find_mdu_opportunity(sr_class, program_name):
    method_name_list = []
    field_name_list = []
    all_statement = []
    chance_statement_list = []
    result_list = []


    field_name_list = list(map(lambda x: x.field_name, sr_class.field_list))
    method_name_list = list(map(lambda x: x.method_name, sr_class.method_list))

    for method in sr_class.method_list:
        method_loc = method.get_method_LOC()
        if method_loc < 5:
            continue
        method_all_statement_list = method.get_all_statement()
        cs_wait_l = get_statement_MDU(method_all_statement_list)

        for statement in cs_wait_l:
            method_name = extract_method_name_from_statement(statement)
            if method_name in method_name_list and method_name != method.method_name:
                index = method_name_list.index(method_name)
                if sr_class.method_list[index].get_method_LOC() > 3:
                    if check_lock_var(method, sr_class.method_list[index], field_name_list) is True:
                        continue
                    new_fix_object = FixObject(
                        target_sr_class=sr_class,
                        program_name=program_name,
                        target_method=method,
                        copy_source_method=sr_class.method_list[index],
                        statement=statement
                    )
                    result_list.append(new_fix_object)


        # all_statement.extend(method.get_all_statement())
    return result_list

def get_statement_param(statement):
    result = []
    start_index = statement.word_list.index("(")
    end_index = statement.word_list.index(")")

    for i in range(start_index+1, end_index):
        if statement.word_list[i] != ",":
            result.append(statement.word_list[i])
    return result

def get_statement_VMU(statement_list):
    # TODO: Required Complete pattern
    pattern = r"^\w+\s\(+"
    result = []
    for statement in statement_list:
        if "." in statement.word_list:
            continue
        if type(statement) == SRStatement:
            sl = statement.to_string().split(" = ")
            if len(sl) == 2:
                # print(sl[1])
                match_obj = re.match(pattern, sl[1], re.M | re.I)
                if match_obj is not None:
                    result.append(statement)

    return result

def find_vmu_opportunity(sr_class, program_name):
    method_name_list = []
    field_name_list = []
    all_statement = []
    chance_statement_list = []
    result_list = []


    field_name_list = list(map(lambda x: x.field_name, sr_class.field_list))
    method_name_list = list(map(lambda x: x.method_name, sr_class.method_list))

    for method in sr_class.method_list:
        method_loc = method.get_method_LOC()
        if method_loc < 5:
            continue
        method_all_statement_list = method.get_all_statement()
        cs_wait_l = get_statement_VMU(method_all_statement_list)

        for statement in cs_wait_l:
            sl = statement.to_string().split(" = ")
            method_name = ""
            if len(sl) == 2:
                split_l = sl[1].split("(")
                method_name = split_l[0].strip()

            if method_name in method_name_list and method_name != method.method_name:
                index = method_name_list.index(method_name)
                if sr_class.method_list[index].get_method_LOC() > 3:
                    if check_lock_var(method, sr_class.method_list[index], field_name_list) is True:
                        continue
                    new_fix_object = FixObject(
                        target_sr_class=sr_class,
                        program_name=program_name,
                        target_method=method,
                        copy_source_method=sr_class.method_list[index],
                        statement=statement
                    )

                    result_list.append(new_fix_object)


        # all_statement.extend(method.get_all_statement())
    return result_list

def get_statement_MDU(statement_list):
    # TODO: Required Complete pattern
    pattern = r"^\w+\s\(+"
    result = []
    for statement in statement_list:
        if "." in statement.word_list:
            continue
        match_obj = re.match(pattern, statement.to_string(), re.M | re.I)
        if match_obj is not None:
            result.append(statement)
    return result

def extract_method_name_from_statement(statement):
    split_l = statement.to_string().split("(")
    return split_l[0].strip()

def check_lock_var(method1, method2, field_name_list):
    method1_all_var = method1.get_all_local_var()
    method2_all_var = method2.get_all_local_var()
    for var in method1_all_var:
        if var in method2_all_var and var not in field_name_list:
            return True
    return False

def find_class_by_name(class_name, cls_list):

    for cls in cls_list:
        if cls.class_name == class_name:
            return cls

def gen_move_method():
    project_path = Path(r"C:\Users\zhoun\PycharmProjects\Sceg\test")
    save_path = Path(r"C:\Users\zhoun\PycharmProjects\Sceg\output")
    ast = KASTParse(project_path, "java")
    ast.setup()
    sr_project = ast.do_parse()
    cls_list = []
    cls_name_list = []
    opp_list = []

    for program in sr_project.program_list:
        for sr_class in program.class_list:
            cls_list.append(sr_class)
            cls_name_list.append(sr_class.class_name)

    for program in sr_project.program_list:
        for sr_class in program.class_list:
            field_name_dict = {}
            for field in sr_class.field_list:
                if field.field_type in cls_name_list:
                    field_name_dict[field.field_name] = field.field_type

            for method in sr_class.method_list:
                for param in method.param_list:
                    if param.type in cls_name_list:
                        new_move_opp = {
                            "target": find_class_by_name(param.type, cls_list),
                            "source": sr_class,
                            "method": method.method_name
                        }
                        opp_list.append(new_move_opp)

                all_statement_list = method.get_all_statement()
                for statement in all_statement_list:
                    for word in statement.word_list:
                        if word in field_name_dict.keys():
                            new_move_opp = {
                                "target": find_class_by_name(field_name_dict[word], cls_list),
                                "source": sr_class.class_name,
                                "method": method.method_name,
                            }
                            opp_list.append(new_move_opp)
    print("opp count:", len(opp_list))

    for opp in opp_list:
        source_class = opp["source"]
        target_class = opp["target"]

        for method in opp["source"].method_list:
            if method.method_name == opp["method"]:


def gen_merge_method():
    project_path = Path(r"D:\research\code_corpus\jgrapht\jgrapht-core\src\main\java")
    save_path = Path(r"D:\research\code_corpus\jgrapht_auto")
    ast = KASTParse(project_path, "java")
    ast.setup()
    sr_project = ast.do_parse()
    mdu_do_fix_object_list = []
    vmu_do_fix_object_list = []

    for program in sr_project.program_list:
        for sr_class in program.class_list:

            mdu_opportunity_list = find_mdu_opportunity(sr_class, program_name=program.program_name)
            vmu_opportunity_list = find_vmu_opportunity(sr_class, program_name=program.program_name)
            if len(mdu_opportunity_list) > 0:
                print("phrase program: %s" % str(program.program_name))
                print("phrase class: %s" % str(sr_class.class_name))
                mdu_do_fix_object_list.extend(mdu_opportunity_list)

            if len(vmu_opportunity_list) > 0:
                print("phrase program: %s" % str(program.program_name))
                print("phrase class: %s" % str(sr_class.class_name))
                vmu_do_fix_object_list.extend(vmu_opportunity_list)

    print("MDU %s can be fixed" % str(len(mdu_do_fix_object_list)))
    print("VMU %s can be fixed" % str(len(vmu_do_fix_object_list)))
    print("Do fix generation...")

    field_order = ["file_name", 'class_name', 'lm_method_name', 'copy_source_method', 'label', 'program_name']
    with open(save_path + "codenet01_index.csv", 'w', encoding="utf-8", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, field_order)
        writer.writeheader()
        # for index, fix_object in enumerate(do_fix_object_list):
        #     file_name = "codenet" + "01" + "-" + str(index) + ".java"
        #     writer.writerow(dict(zip(field_order, [file_name, fix_object.target_sr_class.class_name, fix_object.target_method.method_name, fix_object.copy_source_method.method_name, "1", fix_object.program_name])))
        print(f"Total Num:{len(mdu_do_fix_object_list)}")
        # generate_mdu(mdu_do_fix_object_list=mdu_do_fix_object_list, writer=writer, field_order=field_order)
        generate_amu(vmu_do_fix_object_list=vmu_do_fix_object_list, writer=writer, field_order=field_order)

def gen_merge_cls():
    project_path = Path(r"D:\research\code_corpus\jgrapht\jgrapht-core\src\main\java")
    save_path = Path(r"D:\research\code_corpus\jgrapht_auto")
    ast = KASTParse(project_path, "java")
    ast.setup()
    sr_project = ast.do_parse()
    mopp_list = []
    cls_list = []
    method_name_list = []

    for program in sr_project.program_list:
        for sr_class in program.class_list:
            cls_list.append(sr_class)
            for sr_method in sr_class.method_list:
                method_name_list.append(sr_method.method_name)

    for program in sr_project.program_list:
        for index, sr_class in enumerate(program.class_list):
            ilc, ilcr = check_ilc_merge_oppturnity(sr_class, cls_list)
            if len(ilc) > 0:
                mopp_list.extend(ilc)

            ihc = check_ihc_merge_oppturnity(sr_class, cls_list)
            if len(ihc) > 0:
                mopp_list.extend(ihc)
    print("opp count:", len(mopp_list))

    for index, opp in enumerate(mopp_list):
        gen_lc_opp(opp, index, save_path)

def generate_mdu(mdu_do_fix_object_list, writer, field_order):
    total_num = len(mdu_do_fix_object_list)
    finish_num = 0
    for index, fix_object in enumerate(mdu_do_fix_object_list):

        new_param_list = get_statement_param(fix_object.statement)
        old_param_list = []
        for p in fix_object.copy_source_method.param_list:
            old_param_list.append(p.name)

        sr_class = fix_object.target_sr_class
        if sr_class.class_name == "Metaphone":
            continue
        for method in sr_class.method_list:
            if method.id == fix_object.target_method.id:
                nstm = fix_object.copy_source_method.replace_all_param(
                    old_param_list=old_param_list,
                    new_param_list=new_param_list
                )

                # print("============================")
                # for sd in nstm:
                #     print(sd.to_string())
                # print("============================")

                method.replace_statement(
                    statement_id=fix_object.statement.id,
                    new_statement_list=nstm
                )

        sr_class_gen = copy.deepcopy(sr_class)
        sr_class_gen.class_name = sr_class.class_name + str(index)

        file_name = sr_class.class_name + ".java"
        writer.writerow(dict(zip(field_order, [file_name, sr_class_gen.class_name,
                                               fix_object.target_method.method_name,
                                               fix_object.copy_source_method.method_name, "1",
                                               fix_object.program_name])))

        save_file(
            text=sr_class_gen.to_string(),
            class_name=sr_class_gen.class_name
        )
        finish_num += 1
        print("mdu {}/{} has been finished save".format(finish_num, total_num))

def generate_amu(vmu_do_fix_object_list, writer, field_order):
    total_num = len(vmu_do_fix_object_list)
    finish_num = 0
    for index, fix_object in enumerate(vmu_do_fix_object_list):
        # print("fobs")
        # print(fix_object.statement.to_string())
        new_param_list = get_statement_param(fix_object.statement)
        old_param_list = []
        for p in fix_object.copy_source_method.param_list:
            old_param_list.append(p.name)

        sr_class = fix_object.target_sr_class
        for method in sr_class.method_list:
            if method.id == fix_object.target_method.id:
                nstm = fix_object.copy_source_method.replace_all_param(
                    old_param_list=old_param_list,
                    new_param_list=new_param_list
                )


                # add origin
                sl = fix_object.statement.to_string().split(" = ")
                # print("sl")
                # print(sl)
                if len(sl) == 2:
                    l_sl = sl[0].split(" ")
                    nstm = fix_object.copy_source_method.replace_return_statement(
                        l_s=l_sl
                    )
                    # print("l_sl")
                    # print(l_sl)
                    if len(l_sl) == 2:
                        w_l = []
                        w_l.extend(l_sl)
                        w_l.append("=")
                        # w_l.append('null')
                        w_l.append(';')
                        st = SRStatement(
                            id=fix_object.statement.id,
                            word_list=w_l
                        )
                        nstm.insert(0, st)
                    #
                # print("============================")
                # for sd in nstm:
                #     print(sd.to_string())
                # print("============================")

                method.replace_statement(
                    statement_id=fix_object.statement.id,
                    new_statement_list=nstm
                )

        sr_class_gen = copy.deepcopy(sr_class)
        sr_class_gen.class_name = sr_class.class_name + str(index)

        file_name = sr_class.class_name + ".java"
        writer.writerow(dict(zip(field_order, [file_name, sr_class_gen.class_name,
                                               fix_object.target_method.method_name,
                                               fix_object.copy_source_method.method_name, "1",
                                               fix_object.program_name])))

        save_file(
            text=sr_class_gen.to_string(),
            class_name=sr_class_gen.class_name
        )
        finish_num += 1
        print("vmu {}/{} has been finished save".format(finish_num, total_num))



if __name__ == '__main__':
    # gen_merge_cls()
    # gen_merge_method()
    gen_move_method()