"""Confirm the analysis code and dataset are byte-identical to the files that
produced the paper, using the git blob IDs listed in PROVENANCE.md.

    python verification/check_provenance.py
"""
from __future__ import annotations

import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_blob_id(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")   # git stores LF; Windows checkouts may hold CRLF
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> int:
    with open(os.path.join(ROOT, "PROVENANCE.md"), encoding="utf-8") as f:
        rows = re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{40})` \|$", f.read(), re.M)
    bad = 0
    for rel, want in rows:
        got = git_blob_id(os.path.join(ROOT, rel))
        ok = got == want
        bad += not ok
        print(f"{'OK      ' if ok else 'CHANGED '} {rel}")
    print(f"\n{len(rows) - bad}/{len(rows)} files identical to source commit 1f5883c")
    return 1 if bad or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
