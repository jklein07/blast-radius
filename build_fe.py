"""Pack ftrace.py output into maps/fe.json.gz.

usage: python build_fe.py ftrace.json maps/fe.json.gz
"""
import gzip
import json
import sys

d = json.load(open(sys.argv[1]))
with gzip.open(sys.argv[2], "wt") as fh:
    json.dump(d, fh, separators=(",", ":"))
print({k.split("::")[-1]: (v["nq"], len(v["unstable"]), sum(len(x) for x in v["exec"].values())) for k, v in d.items()})
