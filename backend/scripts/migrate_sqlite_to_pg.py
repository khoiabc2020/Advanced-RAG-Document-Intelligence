import sqlite3
import psycopg
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

def vector_literal(vec):
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

scon = sqlite3.connect("data/ragapp.db")
scon.row_factory = sqlite3.Row
scur = scon.cursor()

pcon = psycopg.connect("postgresql://document_rag:document_rag@localhost:5433/document_rag")
pcur = pcon.cursor()

# 1. users, workspaces, documents
# 2. document_jobs
# 3. workspace_sources
# 4. conversations
# 5. messages
# 6. document_chunks

tables_simple = [
    "users",
    "workspaces",
    "documents",
    "document_jobs",
    "workspace_sources",
    "conversations",
    "messages",
]

for table in tables_simple:
    rows = scur.execute(f"SELECT * FROM {table}").fetchall()
    print(f"Migrating table {table}: {len(rows)} rows...", flush=True)
    if not rows:
        continue
    
    col_names = [d[0] for d in scur.description]
    # Check PG cols
    pcur.execute(f"SELECT column_name, data_type, is_generated FROM information_schema.columns WHERE table_name = '{table}'")
    pg_cols = {r[0]: (r[1], r[2]) for r in pcur.fetchall()}
    
    valid_cols = [c for c in col_names if c in pg_cols and pg_cols[c][1] != 'ALWAYS']
    placeholders = ", ".join(["%s"] * len(valid_cols))
    cols_str = ", ".join(valid_cols)
    
    for row in rows:
        vals = []
        for col in valid_cols:
            v = row[col]
            dtype = pg_cols[col][0]
            if dtype == "boolean" and v is not None:
                vals.append(bool(v))
            else:
                vals.append(v)
                
        pcur.execute(
            f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
            vals,
        )
    pcon.commit()
    print(f"   Done {table}!", flush=True)

# Now document_chunks specifically
rows = scur.execute("SELECT * FROM document_chunks").fetchall()
print(f"Migrating document_chunks: {len(rows)} rows...", flush=True)
chunk_count = 0
for row in rows:
    emb_json = row["embedding_json"]
    if emb_json:
        vec = json.loads(emb_json)
        vec_str = vector_literal(vec)
    else:
        vec_str = None
    
    pcur.execute(
        """
        INSERT INTO document_chunks (
            id, document_id, owner_id, chunk_index, content, 
            page_number, parent_content, chunk_type, metadata_json, 
            embedding, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, 
            %s::vector, %s
        ) ON CONFLICT (id) DO UPDATE SET 
            embedding = EXCLUDED.embedding,
            parent_content = EXCLUDED.parent_content,
            metadata_json = EXCLUDED.metadata_json
        """,
        (
            row["id"],
            row["document_id"],
            row["owner_id"],
            row["chunk_index"],
            row["content"],
            row["page_number"],
            row["parent_content"],
            row["chunk_type"],
            row["metadata_json"],
            vec_str,
            row["created_at"],
        )
    )
    chunk_count += 1
    if chunk_count % 100 == 0:
        print(f"   Migrated {chunk_count}/{len(rows)} chunks...", flush=True)

pcon.commit()
print(f"   Done document_chunks! Total: {chunk_count}", flush=True)

# Verify count in PG
pcur.execute("SELECT COUNT(*) FROM document_chunks")
pg_total = pcur.fetchone()[0]
print(f"Verification: PostgreSQL now has {pg_total} chunks!", flush=True)

pcur.close()
pcon.close()
scon.close()
