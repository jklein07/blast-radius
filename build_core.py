"""Write maps/core.json: the sqlglot modules `import sqlglot` runs at the
pinned commit (with unittest loaded first, as in the selector).

usage: python build_core.py /opt/sqlglot maps/core.json
"""

import json
import sys
import unittest  # noqa: F401  (sqlglot checks for it while importing)

src, dst = sys.argv[1:3]
sys.path.insert(0, src)
before = set(sys.modules)
import sqlglot  # noqa: E402,F401

mods = sorted(m for m in set(sys.modules) - before if m == "sqlglot" or m.startswith("sqlglot."))
with open(dst, "w") as fh:
    json.dump(mods, fh, indent=0)
print(len(mods), "modules", file=sys.stderr)
