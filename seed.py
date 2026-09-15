import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "fridge-recipes"

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

EMBEDDING_DIMENSION = 384

if not pc.has_index(INDEX_NAME):
    print("Creating Pinecone index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Waiting for index to become ready...")
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
else:
    print("Pinecone index already exists.")

index = pc.Index(INDEX_NAME)

sample_recipes = [
    {
        "id": "r1",
        "title": "Simple Omelet",
        "text": """
        Title: Simple Omelet

        Ingredients:
        eggs
        milk
        cheddar cheese
        salt
        butter

        Instructions:
        Whisk eggs and milk.
        Melt butter in a pan.
        Pour the egg mixture into the pan.
        Add cheese.
        Fold the omelet and serve.
        """
    },
    {
        "id": "r2",
        "title": "Tomato Basil Pasta",
        "text": """
        Title: Tomato Basil Pasta

        Ingredients:
        pasta
        tomatoes
        garlic
        olive oil
        basil

        Instructions:
        Boil pasta.
        Saute garlic and tomatoes in olive oil.
        Add cooked pasta.
        Mix everything together.
        Top with fresh basil.
        """
    },
    {
        "id": "r3",
        "title": "Garlic Tomato Toast",
        "text": """
        Title: Garlic Tomato Toast

        Ingredients:
        bread
        tomatoes
        garlic
        olive oil
        salt

        Instructions:
        Toast the bread.
        Chop the tomatoes.
        Mix tomatoes with garlic and olive oil.
        Add salt.
        Place the tomato mixture on the toast.
        """
    },
    {
        "id": "r4",
        "title": "Cheese Pasta",
        "text": """
        Title: Easy Cheese Pasta

        Ingredients:
        pasta
        cheddar cheese
        milk
        butter
        salt

        Instructions:
        Boil pasta.
        Melt butter.
        Add milk and cheese.
        Stir until creamy.
        Add cooked pasta.
        Season with salt and serve.
        """
    }
]

vectors = []

for recipe in sample_recipes:
    print(f"Embedding: {recipe['title']}")
    vector = embedder.encode(recipe["text"]).tolist()
    vectors.append({
        "id": recipe["id"],
        "values": vector,
        "metadata": {
            "title": recipe["title"],
            "recipe_text": recipe["text"]
        }
    })

index.upsert(vectors=vectors)

print()
print("====================================")
print("Pinecone database successfully seeded")
print("====================================")
print(f"Recipes uploaded: {len(vectors)}")
