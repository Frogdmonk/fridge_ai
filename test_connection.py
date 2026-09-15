import os
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq

load_dotenv()

pinecone_key = os.getenv("PINECONE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if not pinecone_key:
    raise ValueError("PINECONE_API_KEY is missing")

if not groq_key:
    raise ValueError("GROQ_API_KEY is missing")

print("API keys found!")

pc = Pinecone(api_key=pinecone_key)
print("Pinecone connected!")
print(pc.list_indexes())

client = Groq(api_key=groq_key)
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "Reply with exactly: Groq connection successful"}
    ]
)

print(response.choices[0].message.content)