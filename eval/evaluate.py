import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foundry_local_sdk import FoundryLocalManager, Configuration
from chunking import chunk_general
from loaders import extract_text
from main import cosine_similarity

EMBED_MODEL = "qwen3-embedding-0.6b-generic-cpu:1"
INSTRUCT = "Instruct: Given a question, retrieve the passage that answers it\nQuery: "
CHUNK_SIZES = [400, 800, 1200]
QUESTIONS_PATH = "eval/questions.json"


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def get_client():
    config = Configuration(app_name="rag_assistant")
    manager = FoundryLocalManager(config)
    manager.start_web_service()
    variant = manager.catalog.get_model_variant(EMBED_MODEL)
    variant.load()
    return variant.get_embedding_client()


def embed(client, text):
    return client.generate_embedding(text).data[0].embedding


def rank(query_vec, chunks, chunk_vecs):
    scored = [(cosine_similarity(query_vec, v), c) for c, v in zip(chunks, chunk_vecs)]
    scored.sort(key=lambda x: -x[0])
    return scored


def load_questions():
    if not os.path.exists(QUESTIONS_PATH):
        sys.exit(f"HATA: {QUESTIONS_PATH} bulunamadı. Önce soruları bu dosyaya yaz.")
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    if not questions:
        sys.exit("HATA: questions.json boş.")
    return questions


def load_text(paths):
    texts, seen = [], []
    for path in paths:
        if not os.path.exists(path):
            sys.exit(f"HATA: dosya bulunamadı -> {path}")
        with open(path, "rb") as f:
            t = extract_text(os.path.basename(path), f.read())

        parts = re.split(r"(?<=[.!?])\s+", t)
        sents = {norm(s) for s in parts if len(s) > 40}
        dup = next(
            (p for p, ss in seen if sents and len(sents & ss) / len(sents) > 0.8),
            None,
        )
        if dup:
            print(f"UYARI: {path} içeriği {dup} ile büyük ölçüde aynı, atlandı.")
            continue
        seen.append((path, sents))
        texts.append(t)
    return "\n\n".join(texts)


def main():
    if len(sys.argv) < 2:
        sys.exit('Kullanım: python eval/evaluate.py "dosya1.pdf" "dosya2.pdf" ...')

    questions = load_questions()
    text = load_text(sys.argv[1:])
    print(f"Toplam {len(text)} karakter, {len(questions)} soru.")

    # Cevap anahtarı kontrolü: snippet metinde gerçekten geçiyor mu?
    full = norm(text)
    for q in questions:
        if q["snippet"] and norm(q["snippet"]) not in full:
            print(f"UYARI: snippet metinde bulunamadı -> {q['question']}")

    client = get_client()

    print("Sorgular vektörleniyor...")
    qvecs = {
        False: [embed(client, q["question"]) for q in questions],
        True: [embed(client, INSTRUCT + q["question"]) for q in questions],
    }

    rows = []
    details = []
    for size in CHUNK_SIZES:
        chunks = chunk_general(text, chunk_size=size, overlap=size // 5)
        print(f"chunk_size={size}: {len(chunks)} parça vektörleniyor...")
        chunk_vecs = [embed(client, c) for c in chunks]

        for use_instruct in (False, True):
            hit1 = hit3 = n = 0
            rr = 0.0
            pos_scores, neg_scores = [], []

            for q, qv in zip(questions, qvecs[use_instruct]):
                ranked = rank(qv, chunks, chunk_vecs)
                top_score = ranked[0][0]
                qtype = q.get("type", "")

                if q["snippet"] is None:
                    neg_scores.append(top_score)
                    details.append([size, "evet" if use_instruct else "hayır", qtype,
                                    q["question"], "", f"{top_score:.3f}"])
                    continue

                n += 1
                snip = norm(q["snippet"])
                pos = next((i for i, (_, c) in enumerate(ranked, 1) if snip in norm(c)), None)
                if pos == 1:
                    hit1 += 1
                if pos is not None and pos <= 3:
                    hit3 += 1
                if pos is not None:
                    rr += 1.0 / pos
                pos_scores.append(top_score)
                details.append([size, "evet" if use_instruct else "hayır", qtype,
                                q["question"], pos if pos else "yok", f"{top_score:.3f}"])

            nan = float("nan")
            rows.append({
                "size": size,
                "instruct": "evet" if use_instruct else "hayır",
                "chunks": len(chunks),
                "top1": hit1 / n if n else nan,
                "top3": hit3 / n if n else nan,
                "mrr": rr / n if n else nan,
                "pos_mean": sum(pos_scores) / len(pos_scores) if pos_scores else nan,
                "neg_max": max(neg_scores) if neg_scores else nan,
            })

    lines = [
        "| Chunk boyutu | Talimat | Parça sayısı | Top-1 | Top-3 | MRR | Ort. skor (cevaplı) | Maks. skor (cevapsız) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['instruct']} | {r['chunks']} | {r['top1']:.2f} | "
            f"{r['top3']:.2f} | {r['mrr']:.2f} | {r['pos_mean']:.2f} | {r['neg_max']:.2f} |"
        )
    table = "\n".join(lines)
    print("\n" + table)

    with open("eval/results.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")

    with open("eval/details.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_size", "talimat", "tip", "soru", "dogru_chunk_sirasi", "top1_skor"])
        w.writerows(details)

    # Ekranda hızlı özet: 800 karakter + talimat için cevapsız soru skorları ve kaçıranlar
    print("\n--- Cevapsız soruların en yüksek skoru (chunk 800, talimat: evet) ---")
    for d in details:
        if d[0] == 800 and d[1] == "evet" and d[4] == "":
            print(f"  {d[5]}  [{d[2]}]  {d[3]}")

    print("\n--- Doğru chunk'ın ilk üçe girmediği sorular (chunk 800, talimat: evet) ---")
    for d in details:
        if d[0] == 800 and d[1] == "evet" and d[4] != "" and (d[4] == "yok" or d[4] > 3):
            print(f"  sıra={d[4]}  skor={d[5]}  [{d[2]}]  {d[3]}")

    print("\n--- Doğru chunk'ın ilk sırada olmadığı ama ilk üçte olduğu sorular (chunk 800, talimat: evet) ---")
    for d in details:
        if d[0] == 800 and d[1] == "evet" and d[4] not in ("", "yok") and 1 < d[4] <= 3:
            print(f"  sıra={d[4]}  skor={d[5]}  [{d[2]}]  {d[3]}")

    print("\nSonuçlar eval/results.md ve eval/details.csv dosyalarına yazıldı.")


if __name__ == "__main__":
    main()