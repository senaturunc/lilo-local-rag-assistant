import itertools
import re
import sys


def sents(path):
    t = open(path, encoding="utf-8").read()
    parts = re.split(r"(?<=[.!?])\s+", t)
    return {re.sub(r"\s+", " ", s.lower()).strip() for s in parts if len(s) > 40}


files = sys.argv[1:]
S = {f: sents(f) for f in files}
for a, b in itertools.combinations(files, 2):
    common = len(S[a] & S[b])
    small = min(len(S[a]), len(S[b]))
    print(f"{a} <-> {b}: ortak cümle {common}, küçük dosyanın %{100 * common / small:.0f}'i")
    