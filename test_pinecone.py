import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("fridge-recipes")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

query = "chicken garlic lemon"

print(f"Searching for: {query}")

query_vector = embedder.encode(query).tolist()

results = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True
)

print("\nRetrieved recipes:\n")

for match in results["matches"]:
    print("--------------------------------")
    print("Score:", match["score"])
    print("Title:", match["metadata"]["title"])
