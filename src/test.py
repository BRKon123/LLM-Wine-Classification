from ai_clients.openai_helper import OpenAIHelper
from dotenv import load_dotenv
from data_stores.chroma_stage_store import ChromaStageStore
import os

load_dotenv()

# Initialize the AI helper with the API key from environment variables
ai_helper = OpenAIHelper(api_key=os.getenv("ARTIFICIAL_API_KEY"))

# Make a simple test call
messages = [
    {"role": "user", "content": "Say hello in a formal way"}
]

try:
    response = ai_helper.create_chat_completion(messages=messages)
    print("AI Response:", response)
except Exception as e:
    print(f"Error occurred: {e}")




# Initialize store with 'description' as the content key
store = ChromaStageStore("my_collection", content_key="description")

store.delete_all_embeddings()

# Create test documents with metadata
test_documents = [
    {
        "title": "Introduction to AI",
        "description": "This is a comprehensive guide about artificial intelligence and its applications",
        "author": "John Doe",
        "date": "2024-03-15"
    },
    {
        "title": "Machine Learning Basics",
        "description": "This document discusses fundamental concepts of machine learning and neural networks",
        "author": "Jane Smith",
        "date": "2024-03-16"
    },
    {
        "title": "Deep Learning",
        "description": "An exploration of deep learning architectures and their implementations",
        "author": "John Doe",
        "date": "2024-03-17"
    }
]

# Add documents
for i, doc in enumerate(test_documents):
    store.add_document(f"doc{i+1}", doc)

# Test retrieval functions
print("\nTesting similar documents retrieval:")
similar = store.get_similar_documents("Tell me about artificial intelligence", n_results=2)
print("\nTop 2 similar documents:")
for doc in similar:
    print(f"\nTitle: {doc['title']}")
    print(f"Description: {doc['description']}")
    print(f"Author: {doc['author']}")

print("\nTesting most similar document retrieval:")
most_similar = store.get_most_similar_document("Tell me about neural networks")
if most_similar:
    print(f"\nMost similar document:")
    print(f"Title: {most_similar['title']}")
    print(f"Description: {most_similar['description']}")
    print(f"Author: {most_similar['author']}")

# Test direct document retrieval
print("\nTesting direct document retrieval:")
doc = store.get_document("doc1")
if doc:
    print(f"\nRetrieved document 1:")
    print(f"Title: {doc['title']}")
    print(f"Description: {doc['description']}")
    print(f"Author: {doc['author']}")

# Test getting all documents
print("\nTesting get all documents:")
all_docs = store.get_all_documents()
print(f"\nTotal documents in store: {len(all_docs)}")