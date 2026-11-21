"""Minimal unified-diff (git format) parser and applier, standard library only."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str] = field(default_factory=list)  # with leading ' ', '-', '+'

    def changed_old_lines(self) -> list[int]:
        """1-based line numbers in the OLD file that were removed, plus the
        old line just before/after each insertion point (for added lines)."""
        out: list[int] = []
        ln = self.old_start
        for raw in self.lines:
            tag = raw[:1]
            if tag == " ":
                ln += 1
            elif tag == "-":
                out.append(ln)
                ln += 1
            elif tag == "+":
                # insertion between ln-1 and ln
                out.append(max(ln - 1, 1))
                out.append(ln)
        return out


    def changed_new_lines(self) -> list[int]:
        """1-based line numbers in the NEW file that were added, plus the new
        lines around each deletion point."""
        out: list[int] = []
        ln = self.new_start
        for raw in self.lines:
            tag = raw[:1]
            if tag == " ":
                ln += 1
            elif tag == "+":
                out.append(ln)
                ln += 1
            elif tag == "-":
                out.append(max(ln - 1, 1))
                out.append(ln)
        return out


@dataclass
class FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: list[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False
    is_binary: bool = False
    new_mode: str | None = None

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    def changed_lines(self) -> tuple[set, set]:
        old: set = set()
        new: set = set()
        for h in self.hunks:
            old.update(h.changed_old_lines())
            new.update(h.changed_new_lines())
        return old, new

    def added_text(self) -> list[str]:
        return [l[1:] for h in self.hunks for l in h.lines if l.startswith("+")]

    def removed_text(self) -> list[str]:
        return [l[1:] for h in self.hunks for l in h.lines if l.startswith("-")]


def _strip(p: str) -> str | None:
    p = p.strip()
    if "\t" in p:
        p = p.split("\t", 1)[0]
    if p == "/dev/null":
        return None
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape").encode("latin-1").decode("utf-8", "replace")
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    return p


def parse(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    cur: FilePatch | None = None
    hunk: Hunk | None = None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            m = re.match(r"diff --git (\"?a/.*?\"?) (\"?b/.*\"?)$", line)
            old = new = None
            if m:
                old, new = _strip(m.group(1)), _strip(m.group(2))
            cur = FilePatch(old, new)
            files.append(cur)
            hunk = None
            i += 1
            continue
        if hunk is not None and line.startswith("\\"):
            # "\ No newline at end of file" applies to previous line
            if hunk.lines:
                hunk.lines[-1] = hunk.lines[-1] + "\\NONL"
            i += 1
            continue
        if hunk is not None and not _hunk_complete(hunk):
            hunk.lines.append(line if line else " ")
            i += 1
            continue
        if line.startswith("--- ") and (hunk is None or _hunk_complete(hunk)):
            if cur is None:
                cur = FilePatch(None, None)
                files.append(cur)
            cur.old_path = _strip(line[4:])
            if cur.old_path is None:
                cur.is_new = True
            hunk = None
            i += 1
            continue
        if line.startswith("+++ ") and cur is not None and hunk is None:
            cur.new_path = _strip(line[4:])
            if cur.new_path is None:
                cur.is_deleted = True
            i += 1
            continue
        m = HUNK_RE.match(line)
        if m and cur is not None:
            hunk = Hunk(
                int(m.group(1)),
                int(m.group(2)) if m.group(2) is not None else 1,
                int(m.group(3)),
                int(m.group(4)) if m.group(4) is not None else 1,
            )
            cur.hunks.append(hunk)
            i += 1
            continue
        if cur is not None:
            if line.startswith("new file mode"):
                cur.is_new = True
                cur.new_mode = line.split()[-1]
            elif line.startswith("deleted file mode"):
                cur.is_deleted = True
            elif line.startswith("rename from "):
                cur.old_path = line[len("rename from "):]
            elif line.startswith("rename to "):
                cur.new_path = line[len("rename to "):]
            elif line.startswith("new mode "):
                cur.new_mode = line.split()[-1]
            elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
                cur.is_binary = True
        i += 1
    return files


def _hunk_complete(h: Hunk) -> bool:
    old = sum(1 for l in h.lines if l[:1] in (" ", "-"))
    new = sum(1 for l in h.lines if l[:1] in (" ", "+"))
    return old >= h.old_len and new >= h.new_len


def apply_to_text(old_text: str | None, fp: FilePatch) -> str | None:
    if fp.is_deleted:
        return None
    if old_text is None:
        old_text = ""
    src = old_text.split("\n")
    # track trailing newline
    trailing_nl = old_text.endswith("\n")
    if trailing_nl:
        src.pop()
    out: list[str] = []
    pos = 0  # index into src
    new_nonl = None
    for h in fp.hunks:
        old_block = [l[1:].replace("\\NONL", "") if l.endswith("\\NONL") else l[1:] for l in h.lines if l[:1] in (" ", "-")]
        new_block = []
        for l in h.lines:
            if l[:1] in (" ", "+"):
                body = l[1:]
                if body.endswith("\\NONL"):
                    body = body[: -len("\\NONL")]
                    new_nonl = True if l[:1] in (" ", "+") else new_nonl
                new_block.append(body)
        start = h.old_start - 1 if h.old_len > 0 else h.old_start
        if src[start : start + len(old_block)] != old_block:
            found = None
            for delta in range(1, len(src) + 1):
                for s in (start - delta, start + delta):
                    if pos <= s <= len(src) - len(old_block) and src[s : s + len(old_block)] == old_block:
                        found = s
                        break
                if found is not None:
                    break
            if found is None:
                raise ValueError(f"hunk does not apply to {fp.path} at line {h.old_start}")
            start = found
        out.extend(src[pos:start])
        out.extend(new_block)
        pos = start + len(old_block)
    reached_end = pos >= len(src)
    out.extend(src[pos:])
    text = "\n".join(out)
    if reached_end and fp.hunks:
        new_side = [l for l in fp.hunks[-1].lines if l[:1] in (" ", "+")]
        trailing = not (new_side and new_side[-1].endswith("\\NONL"))
    else:
        trailing = trailing_nl
    if out and trailing:
        text += "\n"
    return text


def apply(root: str, files: list[FilePatch]) -> None:
    for fp in files:
        if fp.is_binary:
            continue
        old_p = os.path.join(root, fp.old_path) if fp.old_path else None
        new_p = os.path.join(root, fp.new_path) if fp.new_path else None
        old_text = None
        if old_p and os.path.exists(old_p) and not fp.is_new:
            with open(old_p, encoding="utf-8", newline="") as f:
                old_text = f.read()
        if fp.is_deleted:
            if old_p and os.path.exists(old_p):
                os.remove(old_p)
            continue
        new_text = apply_to_text(old_text, fp) if fp.hunks else old_text
        if old_p and new_p and old_p != new_p and os.path.exists(old_p):
            os.remove(old_p)
        if new_p is None:
            continue
        os.makedirs(os.path.dirname(new_p), exist_ok=True)
        if os.path.islink(new_p) or os.path.exists(new_p):
            os.remove(new_p)
        with open(new_p, "w", encoding="utf-8", newline="") as f:
            f.write(new_text or "")
