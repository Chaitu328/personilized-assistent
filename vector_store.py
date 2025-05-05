import os
import re
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from utils import chunk_for_embeddings

# Define common stopwords
STOPWORDS = {
    'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these', 'those',
    'then', 'just', 'so', 'than', 'such', 'when', 'who', 'how', 'where', 'why', 'is', 'are', 'am', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'to', 'from', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'you', 'your'
}

def create_vector_store(text):
    """
    Create a vector store from the provided text
    
    Args:
        text (str): The text to create a vector store from
        
    Returns:
        FAISS: The created vector store
    """
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    
    # Convert chunks to documents
    docs = [Document(page_content=chunk, metadata={}) for chunk in chunks]
    
    # Create the vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    
    return vector_store

def get_retriever(vector_store, search_kwargs=None):
    """
    Get a retriever from the vector store
    
    Args:
        vector_store: The vector store to get a retriever from
        search_kwargs (dict, optional): Search parameters for the retriever
        
    Returns:
        Retriever: The retriever
    """
    if search_kwargs is None:
        search_kwargs = {"k": 4}  # Default to retrieving 4 documents
        
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    return retriever

def similarity_search(vector_store, query, k=4):
    """
    Perform a similarity search in the vector store
    
    Args:
        vector_store: The vector store to search in
        query (str): The query to search for
        k (int): Number of results to return
        
    Returns:
        list: List of similar documents
    """
    results = vector_store.similarity_search(query, k=k)
    return results

def update_vector_store(vector_store, new_text):
    """
    Update the vector store with new text
    
    Args:
        vector_store: The vector store to update
        new_text (str): The new text to add to the vector store
        
    Returns:
        FAISS: The updated vector store
    """
    # Split new text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(new_text)
    
    # Convert chunks to documents
    docs = [Document(page_content=chunk, metadata={}) for chunk in chunks]
    
    # Add documents to the vector store
    vector_store.add_documents(docs)
    
    return vector_store
