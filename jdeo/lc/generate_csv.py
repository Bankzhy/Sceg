import csv
import os

def generate_csv():
    file_order = ["project", "target_class", "extract_methods", "extract_fields"]
    with open("index.csv", "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, file_order)
        writer.writeheader()
        for f in os.listdir("."):
            if f.endswith(".txt"):
                print(f)
                project_name = f.split(".")[0]
                cls_method_dict = {}
                cls_field_dict = {}
                with open(f, "r") as f:
                    all_lines = f.readlines()
                    for index, line in enumerate(all_lines):
                        if line == "":
                            continue
                        ll = line.split("	")
                        target_class_name = ll[0].split(".")[-1]
                        element_l = ll[1].replace("[", "").replace("]", "")
                        if ll[0].startswith("org"):
                            element_l = element_l.split(", org.")
                        else:
                            element_l = element_l.split(", com.")
                        for e in element_l:
                            entity = e.split("::")[-1]
                            if "(" in entity:
                                method_name = entity.split("(")[0]
                                if target_class_name in cls_method_dict.keys():
                                    cls_method_dict[target_class_name].append(method_name)
                                else:
                                    cls_method_dict[target_class_name] = [method_name]
                            elif " " in entity:
                                field_name = entity.split(" ")[-1]
                                if target_class_name in cls_field_dict.keys():
                                    cls_field_dict[target_class_name].append(field_name)
                                else:
                                    cls_field_dict[target_class_name] = [field_name]
                            else:
                                continue

                for cls in cls_method_dict.keys():
                    extract_methods = ";".join(cls_method_dict[cls])
                    extract_fields = ""
                    if cls in cls_field_dict.keys():
                        extract_fields = ";".join(cls_field_dict[cls])

                    extract_methods = extract_methods.replace("\n", "")
                    extract_fields = extract_fields.replace("\n", "")

                    writer.writerow(dict(zip(file_order,[project_name, cls, extract_methods, extract_fields])))

if __name__ == '__main__':
    generate_csv()
