import csv

rows = list(csv.DictReader(open("eval/details.csv", encoding="utf-8-sig")))
sel = [r for r in rows if r["chunk_size"] == "800" and r["talimat"] == "evet"]

ans = sorted(float(r["top1_skor"]) for r in sel if r["dogru_chunk_sirasi"] != "")
print("Cevaplı sorularda en düşük 5 skor:", ans[:5])
print("Cevaplı sorularda medyan:", ans[len(ans) // 2])