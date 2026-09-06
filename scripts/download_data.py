import os
import time
from datasets import load_dataset

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TARGET = 2400

os.makedirs(RAW_DIR, exist_ok=True)

ds = load_dataset("nu-delta/utkface", split="train", streaming=True)

exist = len(os.listdir(RAW_DIR))
print("already present:", exist, flush=True)


def get_row(iterator):
    while True:
        try:
            return next(iterator)
        except StopIteration:
            raise
        except Exception as exc:
            print("transient error:", type(exc).__name__, "retrying...", flush=True)
            time.sleep(2)


it = iter(ds)
count = exist
while count < TARGET:
    try:
        row = get_row(it)
    except StopIteration:
        print("iterator exhausted, restarting stream", flush=True)
        ds = load_dataset("nu-delta/utkface", split="train", streaming=True)
        it = iter(ds)
        continue
    age = int(row["age"])
    gender = 0 if str(row["gender"]).lower().startswith("m") else 1
    fname = "%d_%d_0.jpg" % (age, gender)
    out = os.path.join(RAW_DIR, fname)
    if os.path.exists(out):
        continue
    row["image"].convert("RGB").save(out, "JPEG", quality=90)
    count += 1
    if count % 400 == 0:
        print("downloaded", count, flush=True)

print("done, total", count)