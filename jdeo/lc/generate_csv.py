import csv
import os

def generate_csv():
    file_order = ["project", "target_class", "extract_methods", "extract_fields"]
    with open("index.csv", "w") as csvfile:
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
                        ll = line.split("	")
                        target_class_name = ll[0].split(".")[-1]
                        element_l = ll[1]
                        print(element_l)


if __name__ == '__main__':
    generate_csv()
