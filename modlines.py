"""Record sqlglot lines executed while each test module is imported (build-time tool).

usage: python modlines.py SQLGLOT_ROOT tests.json OUT_JSON
"""
import json
import os
import subprocess
import sys

root, tests_json, out = os.path.realpath(sys.argv[1]), sys.argv[2], sys.argv[3]
if len(sys.argv) > 4:  # child: import one test module
    sys.path.insert(0, root)
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    import fingerprint

    fingerprint.import_all("sqlglot")
    lines = {}
    pre = root + "/"

    def cb(code, line):
        fn = code.co_filename
        if fn.startswith(pre):
            lines.setdefault(fn[len(pre):], set()).add(line)
        return sys.monitoring.DISABLE

    sys.monitoring.use_tool_id(3, "ml")
    sys.monitoring.register_callback(3, sys.monitoring.events.LINE, cb)
    sys.monitoring.set_events(3, sys.monitoring.events.LINE)
    try:
        __import__(sys.argv[4])
    except Exception as e:  # noqa: BLE001
        print("import failed", e, file=sys.stderr)
    sys.monitoring.set_events(3, 0)
    print(json.dumps({f: sorted(v) for f, v in lines.items() if f.startswith("sqlglot/")}))
    sys.exit(0)

files = sorted({t["file"] for t in json.load(open(tests_json))["tests"]})
res = {}
for f in files:
    m = f[:-3].replace("/", ".")
    p = subprocess.run([sys.executable, __file__, root, tests_json, out, m], capture_output=True, text=True,
                       env=dict(os.environ, SKIP_INTEGRATION="0", PYTHONHASHSEED="0"))
    try:
        res[f] = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        print("failed", f, p.stderr[-500:], file=sys.stderr)
        res[f] = {}
json.dump(res, open(out, "w"))
print("module files", len(res), "with lines", sum(1 for v in res.values() if v), file=sys.stderr)
