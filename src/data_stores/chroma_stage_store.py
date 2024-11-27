import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any

class ChromaStageStore:
    """Class to encapsulate ChromaDB store for handling embeddings."""
    
    def __init__(self, collection_name: str, content_key: str = "content", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize ChromaStageStore.
        
        Args:
            collection_name: Name of the ChromaDB collection
            content_key: Key in the document dictionary that contains the text to embed
            model_name: Name of the sentence transformer model to use
        """
        self.client = chromadb.PersistentClient(path="chroma")
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(model_name)
        self.content_key = content_key

    def add_document(self, document_id: str, document: Dict[str, Any]) -> None:
        """
        Add a document dictionary to the ChromaDB collection with its embedding.
        
        Args:
            document_id: Unique identifier for the document
            document: Dictionary containing document data, must include content_key
        """
        try:
            if self.content_key not in document:
                raise KeyError(f"Document must contain '{self.content_key}' key")
            
            content = document[self.content_key]
            if not content or str(content).strip() == "":
                print(f"Skipping empty document {document_id}")
                return
            
            # Generate embedding from the specified content field
            embedding = self.embedding_model.encode(content).tolist()
            
            # Store the entire document as metadata
            self.collection.add(
                ids=[document_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[document]  # Store full document as metadata
            )
                
        except Exception as e:
            print(f"Error processing document {document_id}: {str(e)}")

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Returns:
            The complete document dictionary or None if not found
        """
        results = self.collection.get(
            ids=[document_id],
            include=["metadatas"]
        )
        
        return results['metadatas'][0] if results['metadatas'] else None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieve all documents in the collection.
        
        Returns:
            List of document dictionaries
        """
        results = self.collection.get(
            include=["metadatas"]
        )
        return results['metadatas'] if results['metadatas'] else []

    def get_similar_documents(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents most similar to the query text.
        
        Args:
            query_text: Text to compare against
            n_results: Number of similar documents to return
            
        Returns:
            List of document dictionaries ordered by similarity
        """
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas"]
        )
        
        return results['metadatas'][0] if results['metadatas'][0] else []

    def get_most_similar_document(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Get the single most similar document to the query text.
        
        Returns:
            Most similar document dictionary or None if no documents exist
        """
        similar_docs = self.get_similar_documents(query_text, n_results=1)
        return similar_docs[0] if similar_docs else None

    def delete_all_embeddings(self) -> None:
        """Delete all embeddings in the collection."""
        collection_name = self.collection.name
        self.client.delete_collection(name=collection_name)
        self.collection = self.client.get_or_create_collection(name=collection_name)