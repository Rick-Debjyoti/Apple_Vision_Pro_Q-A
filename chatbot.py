import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from transformers import AutoTokenizer, AutoModel

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key, environment='us-east-1')
index_name = "apple-vision-pro"

index = pc.Index(index_name)

# Initialize OpenAI with Langchain
llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Initialize Langchain with Pinecone and OpenAI
embeddings_model = HuggingFaceEmbeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings_model)
retriever = docsearch.as_retriever()

# Create the RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    llm=llm
)

def get_response(query):
    return chain.run(query)

def get_sales_response(query, user_persona=None):
    # Prompt customization for sales response
    prompt = f"Act as a sales agent for the Apple Vision Pro. Answer the following query: {query}"
    if user_persona:
        prompt += f" Consider the user’s persona: {user_persona}"
    
    # Use Langchain's OpenAI for sales response
    response = llm(prompt)
    return response
