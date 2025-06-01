import csv
import os


def generate_csv():
    file_order = ["project", "source_class", "source_method", "target_class", "source_method_identifier"]
    with open("index.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, file_order)
        writer.writeheader()
        for f in os.listdir("."):
            if f.endswith(".txt"):
                print(f)
                project_name = f.split(".")[0]
                with open(f, "r") as f:
                    all_lines = f.readlines()
                    for index, line in enumerate(all_lines):

                        ll = line.split("	")
                        source_entity = ll[1]
                        target_entity = ll[2]

                        source_entity_l = source_entity.split("::")
                        source_class_l = source_entity_l[0].split(".")
                        source_class_name = source_class_l[-1]
                        source_method = source_entity_l[1]
                        source_method_name = source_method.split("(")[0]
                        source_method_identifier = source_method.split(":")[0]
                        target_class_name = target_entity.split(".")[-1]
                        writer.writerow(dict(zip(file_order, [project_name, source_class_name, source_method_name, target_class_name, source_method_identifier])))







if __name__ == '__main__':
    generate_csv()
