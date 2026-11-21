"""Build-time: fingerprint the pristine sqlglot import state. usage: mksnap.py ROOT OUT"""
import json
import os
import sys
import unittest  # noqa: F401  sqlglot turns on strict argument checks when unittest is loaded at import; tests always run that way

root = sys.argv[1]
sys.path.insert(0, root)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
import fingerprint  # noqa: E402

snap = fingerprint.snapshot("sqlglot")
with open(sys.argv[2], "w") as f:
    json.dump(snap, f)
print("classes", len(snap["classes"]), "errors", snap["errors"], file=sys.stderr)
# entries of dict-valued class attributes, by key (selector: which keys a change touched)
dk = fingerprint.dict_snapshot("sqlglot")
with open(os.path.splitext(sys.argv[2])[0] + "dk.json", "w") as f:
    json.dump(dk, f, separators=(",", ":"))
print("dict attrs", len(dk), "entries", sum(len(v) for v in dk.values()), file=sys.stderr)
