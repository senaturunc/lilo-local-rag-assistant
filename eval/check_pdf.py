import sys
import pymupdf

d = pymupdf.open(sys.argv[1])
t = "".join(p.get_text() for p in d)
print(len(d), "sayfa,", len(t), "karakter")
print(t[:300])