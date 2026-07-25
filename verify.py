import database

print("SQLite veri tabanındaki veriler okunuyor...")
chunks = database.get_all_chunks()
print(f"Veri tabanındaki toplam parça (chunk) sayısı: {len(chunks)}")
print("-" * 60)

for chunk in chunks:
    print(f"ID: {chunk['id']}")
    print(f"Dosya Adı: {chunk['filename']}")
    print(f"İçerik:\n{chunk['content']}")
    print(f"Vektör Boyutu (Embedding Length): {len(chunk['embedding'])}")
    print(f"Vektörün ilk 5 değeri: {chunk['embedding'][:5]}...")
    print("-" * 60)
