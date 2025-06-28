import os, chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY!")

class EmbedWrapper:
    def __init__(self):
        self.model = OpenAIEmbeddings(model="text-embedding-3-small")
    def __call__(self, input):
        return self.model.embed_documents(input)
    def name(self):
        return "openai-1536"

embedder = EmbedWrapper()
client   = chromadb.PersistentClient(path="chroma_index")
col      = client.get_collection("dbqs", embedding_function=embedder)
oai      = OpenAI()

app = FastAPI()

class ChatIn(BaseModel):
    user: str

@app.get("/")
async def root():
    return {"msg": "DOC backend up"}

@app.post("/practice")
async def practice(inp: ChatIn):
    q_vec  = embedder([inp.user])[0]
    hits   = col.query(query_embeddings=[q_vec], n_results=3)
    context = "\n---\n".join(hits["documents"][0])

    messages = [
        {"role":"system","content":"You are D.O.C. Practice Agent."},
        {"role":"system","content":context},
        {"role":"user","content":inp.user},
    ]
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return {"answer": resp.choices[0].message.content}

