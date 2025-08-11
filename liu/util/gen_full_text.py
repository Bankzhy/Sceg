from common import project_path_dict
from sitter.kast2core import KASTParse


def camel_case_split(str):
    if len(str) == 0:
        return ""
    words = [[str[0]]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]

def under_case_split(w):
    ws = w.split("_")
    return ws

def split_token(word):

    if "_" in word:
        result = under_case_split(word)
    else:
        result = camel_case_split(word)
    return result

def run():
    results = []
    for key in project_path_dict:
        print("parsing: %s", key)
        project_path = project_path_dict[key]
        ast = KASTParse(project_path, "java")
        ast.setup()
        sr_project = ast.do_parse()
        for program in sr_project.program_list:
            for sr_class in program.class_list:
                class_name_txt = split_token(sr_class.class_name)
                class_name_txt = " ".join(class_name_txt)
                results.append(class_name_txt.lower())

                for f in sr_class.field_list:
                    field_name_txt = split_token(f.field_name)
                    field_name_txt = " ".join(field_name_txt)
                    results.append(field_name_txt.lower())
                for m in sr_class.method_list:
                    method_name_txt = split_token(m.method_name)
                    method_name_txt = " ".join(method_name_txt)
                    results.append(method_name_txt.lower())
    results= list(set(results))
    with open("mn_full.txt", "w", encoding="utf-8") as f:
        for w in results:
            f.write(w)
            f.write("\n")
        f.close()

if __name__ == '__main__':
    run()