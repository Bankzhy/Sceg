from pathlib import Path

from sitter.ast2core import ASTParse
from sitter.kast2core import KASTParse


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
                        print("new merge opp:")
                        print(new_merge_opp)
                        print("="*100)
                        result_list.append(new_merge_opp)
            else:
                new_merge_opp = {
                    "sc": sr_cls,
                    "type": "ilc",
                    "tc": field_cls,
                }
                print("new merge opp:")
                print(new_merge_opp)
                print("=" * 100)
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

def gen_move_method():
    pass

def gen_merge_method():
    pass

def gen_merge_cls():
    project_path = Path(r"D:\research\code_corpus\jgrapht\jgrapht-core\src\main\java")
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
            mopp_list.extend(check_ilc_merge_oppturnity(sr_class, cls_list))
            mopp_list.extend(check_ihc_merge_oppturnity(sr_class, cls_list))
    print("opp count:", len(mopp_list))

if __name__ == '__main__':
    gen_merge_cls()