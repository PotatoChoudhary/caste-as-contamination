"""castelens/utils.py — tiny IO helpers so every experiment saves a result the same way."""
import json
import os
import time
from . import config


def save_result(name, obj):
    obj = {"_saved": time.strftime("%Y-%m-%d %H:%M:%S"), **obj}
    path = os.path.join(config.RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  saved -> {path}")
    return path


def load_result(name):
    with open(os.path.join(config.RESULTS_DIR, f"{name}.json")) as f:
        return json.load(f)


def load_data(fname):
    with open(os.path.join(config.DATA_DIR, fname)) as f:
        return json.load(f)
