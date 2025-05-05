import os
import re
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def answer_question(question, retriever):
    """
    Answer a question based on the retrieved documents
    
    Args:
        question (str): The question to answer
        retriever: The retriever to use for RAG
        
    Returns:
        str: Answer to the question
    """
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(question)
    
    # Format the context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    system_prompt = (
        "You are a helpful educational assistant. Answer the student's question based "
        "on the provided course material. Be concise but thorough, and make sure "
        "your answer is directly relevant to what is being asked. If the question "
        "cannot be answered based on the provided material, say so clearly."
    )
    
    user_prompt = f"Question: {question}\n\nRelevant course material:\n{context}"
    
    # Simple text-based approach to answer questions using the retrieved context
    try:
        # Extract keywords from the question
        keywords = re.findall(r'\b\w+\b', question.lower())
        keywords = [k for k in keywords if k not in {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'to', 'and', 'or', 'but', 'is', 'are', 'what', 'when', 'where', 'why', 'how'}]
        
        # Find sentences in the context that contain keywords
        relevant_sentences = []
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Count keywords in the sentence
            keyword_count = sum(1 for k in keywords if k in sentence.lower())
            if keyword_count > 0:
                relevant_sentences.append((sentence, keyword_count))
        
        # Sort by keyword count
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Construct an answer using the most relevant sentences
        if relevant_sentences:
            answer_components = [s[0] for s in relevant_sentences[:3]]
            answer = " ".join(answer_components)
            return answer
        else:
            return "I couldn't find a specific answer to your question in the provided course materials."
    
    except Exception as e:
        print(f"Error answering question: {e}")
        return f"Sorry, I encountered an error while trying to answer your question: {str(e)}"

def create_qa_chain(retriever):
    """
    Create a RetrievalQA chain using langchain
    
    Args:
        retriever: The retriever to use for RAG
        
    Returns:
        RetrievalQA: The QA chain
    """
    # Use a simpler approach without LLMs
    # Return a function that performs retrieval and basic sentence matching
    # This is a simplified version since we're not using Groq LLMs
    
    def answer_function(query):
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Extract keywords from the question
        keywords = re.findall(r'\b\w+\b', query.lower())
        keywords = [k for k in keywords if k not in {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'to', 'and', 'or', 'but', 'is', 'are', 'what', 'when', 'where', 'why', 'how'}]
        
        # Find sentences in the context that contain keywords
        relevant_sentences = []
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Count keywords in the sentence
            keyword_count = sum(1 for k in keywords if k in sentence.lower())
            if keyword_count > 0:
                relevant_sentences.append((sentence, keyword_count))
        
        # Sort by keyword count
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Construct an answer using the most relevant sentences
        if relevant_sentences:
            answer_components = [s[0] for s in relevant_sentences[:3]]
            answer = " ".join(answer_components)
            return {"result": answer, "source_documents": docs}
        else:
            return {"result": "I couldn't find a specific answer to your question in the provided course materials.", "source_documents": docs}
    
    # Return a callable object that mimics the chain interface
    class SimpleQAChain:
        def __init__(self, answer_func):
            self.answer_func = answer_func
            
        def __call__(self, query):
            return self.answer_func(query)
    
    return SimpleQAChain(answer_function)
