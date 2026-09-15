import os
import ast
import time

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")

INDEX_NAME = "fridge-recipes"
SAMPLE_SIZE = 400
BATCH_SIZE = 64

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("Clearing old data from index...")
try:
    index.delete(delete_all=True)
except Exception as e:
    print("Nothing to clear or delete failed (fine if index was empty):", e)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading CSV...")
df = pd.read_csv("recipe.csv")

df = df.dropna(subset=["Title", "Instructions", "Cleaned_Ingredients"])

df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

print(f"Using {len(df)} recipes after cleaning and sampling.")


def parse_ingredients(raw):
    try:
        items = ast.literal_eval(raw)
        if isinstance(items, list):
            return ", ".join(str(i) for i in items)
    except Exception:
        pass
    return str(raw)


records = []

for i, row in df.iterrows():
    title = str(row["Title"]).strip()
    ingredients = parse_ingredients(row["Cleaned_Ingredients"])
    instructions = str(row["Instructions"]).strip()

    text = f"Title: {title}\n\nIngredients:\n{ingredients}\n\nInstructions:\n{instructions}"

    records.append({
        "id": f"kaggle-{i}",
        "title": title,
        "text": text
    })

print("Embedding and uploading in batches...")

for start in range(0, len(records), BATCH_SIZE):
    batch = records[start:start + BATCH_SIZE]

    texts = [r["text"] for r in batch]
    vectors = embedder.encode(texts).tolist()

    upsert_payload = []
    for r, vec in zip(batch, vectors):
        upsert_payload.append({
            "id": r["id"],
            "values": vec,
            "metadata": {
                "title": r["title"],
                "recipe_text": r["text"][:3000]
            }
        })

    index.upsert(vectors=upsert_payload)
    print(f"Uploaded {start + len(batch)} / {len(records)}")

print()
print("====================================")
print("Kaggle dataset successfully seeded")
print("====================================")
print(f"Total recipes uploaded: {len(records)}")
