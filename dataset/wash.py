
import pymysql

db = pymysql.connect(
    host="47.113.220.80",
    user="root",
    password="Apple3328823%",
    database="sce",
    charset="utf8mb4",  # Use utf8mb4 for full Unicode support
    connect_timeout=5000
)


def filter_extract_class():
    cursor = db.cursor()
    cursor.execute("select * from lc_master where label=1 and reviewer_id=57;")

    for row in cursor.fetchall():
        lc_id = row[0]
        extract_methods = row[4]

        if ", " in extract_methods:

            emn = extract_methods.replace(" ", "")


            query = (r"update lc_master set extract_methods=%s where lc_id=%s;")
            values = (emn, lc_id)
            print(lc_id)
            print(emn)
            print("="*10)
            cursor.execute(query, values)
    db.commit()

if __name__ == '__main__':
    filter_extract_class()