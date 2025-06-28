# ----- embed_dbqs.py -----
import os, re
from pathlib import Path
import chromadb
from langchain_openai import OpenAIEmbeddings

DBQ_DIR   = Path("dbq_library")   # folder that holds your *_DBQ_CLEAN.md files
INDEX_DIR = "chroma_index"        # on-disk vector DB folder

class EmbedWrapper:
    """Adapter so Chroma likes the new OpenAIEmbeddings object."""
    def __init__(self):
        self.model = OpenAIEmbeddings(model="text-embedding-3-small")

    # <-- Chroma now calls __call__(input=[...])
    def __call__(self, input):
        return self.model.embed_documents(input)

    def embed_query(self, text: str):
        return self.model.embed_query(text)

    def name(self):                # Chroma still asks for a string label
        return "openai-1536"

def split_md(md_text: str, chunk_size: int = 800):
    """Very rough splitter: keep headers intact, then wrap full paragraphs."""
    pieces, chunks = re.split(r"\n#+\s+", md_text), []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        while len(p) > chunk_size:
            chunks.append(p[:chunk_size])
            p = p[chunk_size:]
        chunks.append(p)
    return chunks

def build_index():
    embedder = EmbedWrapper()
    client   = chromadb.PersistentClient(path=INDEX_DIR)

    # ── HARD RESET so we never mix old/new dimensions ───────────────
    try:
        client.delete_collection("dbqs")
    except Exception:
        pass

    col = client.get_or_create_collection(
        "dbqs",
        embedding_function=embedder,          # 1 536-dim OpenAI vectors
    )

    for md_file in DBQ_DIR.glob("*_DBQ_CLEAN.md"):
        text   = md_file.read_text(encoding="utf-8")
        chunks = split_md(text)
        for i, chunk in enumerate(chunks):
            col.add(
                ids        =[f"{md_file.stem}_{i}"],
                documents  =[chunk],
                metadatas  =[{"source": md_file.name}],
            )

if __name__ == "__main__":
    build_index()
    print("✅ Index built.")

