import csv

path = "F:/Project/Poly-FEVER/Poly-FEVER.tsv"

with open(path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    print("Columns:", reader.fieldnames)
    for i, row in enumerate(reader):
        print(f"\nRow {i+1}:", dict(row))
        if i >= 2:
            break
