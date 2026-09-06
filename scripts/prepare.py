import csv
import glob
import os
import random

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PREP_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "prepared")
SEED = 42
VAL_SPLIT = 0.2

random.seed(SEED)
os.makedirs(PREP_DIR, exist_ok=True)

rows = []
for path in glob.glob(os.path.join(RAW_DIR, "*.jpg")):
    base = os.path.basename(path)
    try:
        age_s, gen_s, _ = base.split("_")[:3]
        age = int(age_s)
        gender = int(gen_s)
    except ValueError:
        continue
    rows.append((path, age, gender))

random.shuffle(rows)
n_val = int(len(rows) * VAL_SPLIT)
val_rows = rows[:n_val]
train_rows = rows[n_val:]

def write_csv(fname, data):
    with open(os.path.join(PREP_DIR, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "age", "gender"])
        w.writerows(data)

write_csv("train.csv", train_rows)
write_csv("val.csv", val_rows)

males = sum(1 for _, _, g in rows if g == 0)
ages = [a for _, a, _ in rows]
print("total", len(rows), "| train", len(train_rows), "| val", len(val_rows))
print("gender male:female =", males, ":", len(rows) - males)
print("age min/mean/max =", min(ages), round(sum(ages) / len(ages), 1), max(ages))