# ----- main.py -----
import os, chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from embed_dbqs import EmbedWrapper          # uses the same wrapper
from openai import OpenAI, BadRequestError

# === safety check ===
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY before launching!")

# === vector store ===
embedder   = EmbedWrapper()
client     = chromadb.PersistentClient(path="chroma_index")
collection = client.get_collection("dbqs", embedding_function=embedder)

# === OpenAI client (we’ll reuse the same instance) ===
oai = OpenAI()

app = FastAPI()

class Query(BaseModel):
    user: str

@app.post("/practice")
async def practice(q: Query):
    # 1️⃣ embed the incoming question
    q_vec = embedder.embed_query(q.user)

    # 2️⃣ look for the three closest DBQ chunks
    hits  = collection.query(query_embeddings=[q_vec], n_results=3)

    if not hits["documents"][0]:           # <-- nothing matched?
        return {
            "answer": (
                "I don’t have that DBQ in my library yet. "
                "Upload a cleaned copy and I’ll pull the official language."
            )
        }

    # 3️⃣ build a minimal context prompt
    context = "\n\n".join(hits["documents"][0])
    prompt  = (
        "You are the D.O.C. Practice Agent. "
        "Answer ONLY from the VA Disability Benefits Questionnaire excerpt below. "
        "If the excerpt doesn’t cover the topic, coach the veteran on where to "
        "explain it during a real exam, but do NOT add new medical content.\n\n"
        f"{context}\n\nVeteran’s question: {q.user}\nAnswer:"
    )

    # 4️⃣ call OpenAI
    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return {"answer": resp.choices[0].message.content.strip()}
    except BadRequestError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

