import sqlite3
import json

DB_FILE = "rag.db"

def get_connection():
    """SQLite veri tabanina baglanti olusturur."""
    return sqlite3.connect(DB_FILE)

def init_db():
    """Veri tabanini ve gerekli tablolari olusturur."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("Veri tabani basariyla baslatildi ve tablo olusturuldu.")

def save_chunk(filename, content, embedding_vector):
    """Bir metin parçasını (chunk) ve onun vektörünü veri tabanına kaydeder."""
    conn = get_connection()
    cursor = conn.cursor()
    # Vektörü SQLite'da saklayabilmek için JSON string formatına çeviriyoruz
    embedding_json = json.dumps(embedding_vector)
    cursor.execute("""
        INSERT INTO documents (filename, content, embedding)
        VALUES (?, ?, ?)
    """, (filename, content, embedding_json))
    conn.commit()
    conn.close()

def get_all_chunks():
    """Veri tabanındaki tüm kayıtları çeker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    chunks = []
    for row in rows:
        chunks.append({
            "id": row[0],
            "filename": row[1],
            "content": row[2],
            "embedding": json.loads(row[3])  # JSON string'i tekrar sayı listesine (vektöre) çeviriyoruz
        })
    return chunks

def delete_document(filename):
    """Belirli bir dosyaya ait tüm parçaları (vektörleri) veri tabanından siler."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()

def get_unique_filenames():
    """Veri tabanında kayıtlı olan tüm benzersiz dosya isimlerini çeker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT filename FROM documents")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

if __name__ == "__main__":
    init_db()
