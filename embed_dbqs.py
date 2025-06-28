import os, re
from pathlib import Path
import chromadb
from langchain_openai import OpenAIEmbeddings

DBQ_DIR   = Path("dbq_library")
INDEX_DIR = "chroma_index"

class EmbedWrapper:
    def __init__(self):
        self.model = OpenAIEmbeddings(model="text-embedding-3-small")

    # REQUIRED BY CHROMA ≥0.4.16  (param name must be *input*)
    def __call__(self, input):
        return self.model.embed_documents(input)

    def name(self):           # helpful label
        return "openai-1536"

embedder = EmbedWrapper()

def split_md(md: str, chunk=800):
    parts, out = re.split(r"\n###\s+", md), []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > chunk:
            out.append(p[:chunk])
            p = p[chunk:]
        out.append(p)
    return out

def build_index():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY!")

    client = chromadb.PersistentClient(path=INDEX_DIR)
    try:
        client.delete_collection("dbqs")
    except Exception:
        pass

    col = client.get_or_create_collection(
        name="dbqs",
        embedding_function=embedder
    )

    for md in DBQ_DIR.glob("*_DBQ_CLEAN.md"):
        text = md.read_text("utf-8")
        for i, chunk in enumerate(split_md(text)):
            col.upsert(
                ids=[f"{md.stem}_{i}"],
                documents=[chunk],
                metadatas=[{"source": md.name}],
            )
    print("✅ Index built with", col.count(), "chunks")

if __name__ == "__main__":
    build_index()

