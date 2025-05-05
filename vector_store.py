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

# Simple document store class
class SimpleDocStore:
    """A simple document store with basic text-based retrieval"""

    def __init__(self, documents):
        """
        Initialize with a list of Document objects
        
        Args:
            documents (List[Document]): List of Document objects
        """
        self.documents = documents
        
        # Create an index of word frequencies for each document
        self.document_terms = []
        for doc in documents:
            # Tokenize and clean each document
            text = doc.page_content.lower()
            # Split on non-alphanumeric chars and filter out stopwords
            terms = [term for term in re.findall(r'\b\w+\b', text) 
                     if term not in STOPWORDS and len(term) > 2]
            # Count term frequencies
            term_counter = Counter(terms)
            self.document_terms.append(term_counter)
            
    def as_retriever(self, search_kwargs=None):
        """Return self as a retriever-like object"""
        return self
    
    def get_relevant_documents(self, query, k=4):
        """
        Retrieve relevant documents for the query
        
        Args:
            query (str): The query text
            k (int): Number of documents to retrieve
            
        Returns:
            List[Document]: List of relevant documents
        """
        # Extract terms from query
        query_terms = [term for term in re.findall(r'\b\w+\b', query.lower()) 
                     if term not in STOPWORDS and len(term) > 2]
        
        # Calculate similarity score for each document
        scores = []
        for i, doc_terms in enumerate(self.document_terms):
            # Count matching terms
            score = sum(doc_terms[term] for term in query_terms if term in doc_terms)
            scores.append((i, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        top_docs = [self.documents[i] for i, _ in scores[:k]]
        return top_docs
    
    def add_documents(self, documents):
        """
        Add new documents to the store
        
        Args:
            documents (List[Document]): List of Document objects to add
        """
        # Add the documents
        self.documents.extend(documents)
        
        # Update the index for the new documents
        for doc in documents:
            text = doc.page_content.lower()
            terms = [term for term in re.findall(r'\b\w+\b', text) 
                     if term not in STOPWORDS and len(term) > 2]
            term_counter = Counter(terms)
            self.document_terms.append(term_counter)
            
    def similarity_search(self, query, k=4):
        """
        Alias for get_relevant_documents
        """
        return self.get_relevant_documents(query, k)

def create_vector_store(text):
    """
    Create a document store from the provided text
    
    Args:
        text (str): The text to create a document store from
        
    Returns:
        SimpleDocStore: The created document store
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
    
    # Create the document store
    doc_store = SimpleDocStore(docs)
    
    return doc_store

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
    Update the document store with new text
    
    Args:
        vector_store: The document store to update
        new_text (str): The new text to add to the document store
        
    Returns:
        SimpleDocStore: The updated document store
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
    
    # Add documents to the document store
    vector_store.add_documents(docs)
    
    return vector_store
