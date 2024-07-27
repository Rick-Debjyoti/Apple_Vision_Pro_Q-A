import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Create index
index_name = "apple-vision-pro"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws',region='us-east-1')
    )

index = pc.Index(index_name)

# Load Huggingface model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def chunk_text(text, chunk_size=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return chunks


# Read and combine extracted text data
files = ["extracted_text.txt", "web_scraped_text.txt", "youtube_transcript.txt"]
combined_text = ""

for file in files:
    with open(file, "r") as f:
        combined_text += f.read()

# Chunk the combined text
chunks = chunk_text(combined_text, chunk_size=512)

# Embed and upsert chunks into Pinecone
for i, chunk in enumerate(chunks):
    chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
    embeddings = embed_text(chunk_text)
    metadata = {"chunk": i, "text": chunk_text}  
    index.upsert([(str(i), embeddings, metadata)])

print("Data has been successfully embedded and uploaded to Pinecone.")

