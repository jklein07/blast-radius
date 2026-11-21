"""Build-time: repack maps/maps.json.gz into a marshal file of compact byte
arrays, so the selector loads it in milliseconds and decodes only what a
patch touches."""
import array
import gzip
import json
import marshal
import sys

src, dst = sys.argv[1], sys.argv[2]
with gzip.open(src, "rt") as fh:
    m = json.load(fh)


def u16(xs):
    return array.array("H", xs).tobytes()


def i32(xs):
    return array.array("i", xs).tobytes()


m["sets"] = [u16(x) for x in m["sets"]]
m["cov"] = {f: i32([v for ln, sid in d.items() for v in (int(ln), sid)]) for f, d in m["cov"].items()}
files = sorted({f for mod in m["names"].values() for refs in mod.values() for f, _ in refs})
fidx = {f: i for i, f in enumerate(files)}
m["name_files"] = files
m["names"] = {mod: {n: i32([v for f, ln in refs for v in (fidx[f], ln)]) for n, refs in d.items()} for mod, d in m["names"].items()}
with open(dst, "wb") as fh:
    marshal.dump(m, fh)
