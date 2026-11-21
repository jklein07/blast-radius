"""Build-time: fingerprint the pristine sqlglot import state. usage: mksnap.py ROOT OUT"""
import json
import os
import sys

root = sys.argv[1]
sys.path.insert(0, root)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
import fingerprint  # noqa: E402

snap = fingerprint.snapshot("sqlglot")
with open(sys.argv[2], "w") as f:
    json.dump(snap, f)
print("classes", len(snap["classes"]), "errors", snap["errors"], file=sys.stderr)
