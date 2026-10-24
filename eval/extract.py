import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import extract_text

for pdf_path in sys.argv[1:]:
    with open(pdf_path, "rb") as f:
        text = extract_text(os.path.basename(pdf_path), f.read())
    out_path = "eval/" + os.path.splitext(os.path.basename(pdf_path))[0] + "_text.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(text)
    print(f"{out_path}: {len(text)} karakter")