import os
import random
import tarfile

TAR_PATH = os.path.join(os.environ.get("TEMP", "."), "UTKFace.tar.gz")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TARGET = 2400
SEED = 42

random.seed(SEED)
os.makedirs(RAW_DIR, exist_ok=True)

BUCKETS = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 60), (61, 130)]
needed = TARGET // (2 * len(BUCKETS))
selected = {}  # key -> count
count = 0

with tarfile.open(TAR_PATH) as tar:
    for member in tar.getmembers():
        if not member.isfile():
            continue
        name = os.path.basename(member.name)
        try:
            age_s, gen_s = name.split("_")[:2]
            if not age_s.isdigit() or gen_s not in ("0", "1"):
                continue
            age = int(age_s)
        except Exception:
            continue
        for lo, hi in BUCKETS:
            if lo <= age <= hi:
                key = (lo, hi, gen_s)
                if selected.get(key, 0) >= needed:
                    break
                selected[key] = selected.get(key, 0) + 1
                tar.extract(member, RAW_DIR, filter="data")
                count += 1
                break

print("extracted", count, "images to", os.path.abspath(RAW_DIR))