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
        # First, handle special cases like questions about the overall document topic
        question_lower = question.lower()
        
        # Special case handling for general document questions
        if any(phrase in question_lower for phrase in ["what is this pdf about", "what is this document about", 
                "main topic", "which topic", "what topic", "subject of this", "this pdf explain"]):
            
            # For questions about the document's overall topic, take a different approach
            # Analyze the first few paragraphs which typically contain topic information
            paragraphs = context.split('\n\n')
            intro_text = '\n\n'.join(paragraphs[:min(3, len(paragraphs))])
            
            # Extract likely topic indicators from the introduction
            topic_indicators = ["introduction", "overview", "about", "this paper", "this document", 
                              "we present", "discusses", "examines", "explores", "focuses on"]
            
            topic_sentences = []
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', intro_text)
            
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in topic_indicators):
                    topic_sentences.append(sentence)
            
            # If we found specific topic sentences, use them
            if topic_sentences:
                return "This document appears to be about " + " ".join(topic_sentences[:2])
            
            # Otherwise, use the first 2-3 sentences which often state the document's purpose
            else:
                return "Based on the document content, it appears to cover: " + " ".join(sentences[:3])
        
        # Regular question handling continues below
        # Identify question type
        question_starters = {
            "what": "descriptive",
            "who": "entity",
            "when": "temporal",
            "where": "location",
            "why": "reasoning",
            "how": "process",
            "which": "selection",
            "can": "possibility",
            "does": "verification",
            "is": "verification",
            "are": "verification",
            "do": "verification",
            "define": "definition",
            "explain": "explanation",
            "compare": "comparison",
            "contrast": "comparison",
            "list": "enumeration",
            "describe": "descriptive"
        }
        
        # Determine question type
        question_type = "general"
        for starter, q_type in question_starters.items():
            if question_lower.startswith(starter) or f" {starter} " in question_lower:
                question_type = q_type
                break
                
        # Extract meaningful keywords (excluding common words)
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'to', 'and', 'or', 
                       'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'this', 'that', 
                       'these', 'those', 'it', 'its', 'they', 'them', 'their', 'he', 'she', 'him', 'her', 
                       'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose'}
        
        # Extract all words
        all_words = re.findall(r'\b\w+\b', question_lower)
        
        # Filter important keywords
        keywords = [k for k in all_words if k not in common_words and len(k) > 2]
        
        # If we have too few keywords, include some common question words that might be important
        if len(keywords) < 2:
            question_specific_words = ['what', 'when', 'where', 'why', 'how', 'which', 'who']
            for word in all_words:
                if word in question_specific_words and word not in keywords:
                    keywords.append(word)
        
        # 2. Score sentences based on multiple factors
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence.split()) < 5:
                continue
                
            sentence_lower = sentence.lower()
            
            # Multiple scoring factors:
            # a. Keyword matches (basic relevance)
            # b. Keyword density (prefer sentences with higher concentration of keywords)
            # c. Sentence length (prefer reasonable length sentences)
            # d. Question type matching (e.g., "when" questions should match sentences with dates)
            
            # Basic keyword match count
            keyword_count = sum(1 for k in keywords if k in sentence_lower)
            
            # Skip sentences with no keyword matches
            if keyword_count == 0:
                continue
                
            # Calculate keyword density (matches per word)
            word_count = len(sentence.split())
            keyword_density = keyword_count / word_count if word_count > 0 else 0
            
            # Length factor (prefer medium-length sentences)
            length_factor = 1.0
            if 10 <= word_count <= 25:  # Ideal length
                length_factor = 1.5
            elif word_count > 40:  # Too long
                length_factor = 0.7
                
            # Question type matching bonus
            type_match_bonus = 1.0
            if question_type == "temporal" and re.search(r'\b(in|on|during|year|date|when|time|century|decade|period|era|age)\b', sentence_lower):
                type_match_bonus = 2.0
            elif question_type == "location" and re.search(r'\b(in|at|on|near|location|place|where|region|area|country|city|state)\b', sentence_lower):
                type_match_bonus = 2.0
            elif question_type == "entity" and re.search(r'\b(person|people|who|name|individual|group|organization|company)\b', sentence_lower):
                type_match_bonus = 2.0
            elif question_type == "reasoning" and re.search(r'\b(because|since|reason|cause|effect|result|due to|why|therefore)\b', sentence_lower):
                type_match_bonus = 2.0
            elif question_type == "definition" and re.search(r'\b(is|are|refers to|defined as|means|definition)\b', sentence_lower):
                type_match_bonus = 2.0
            
            # Calculate final score
            final_score = keyword_count * keyword_density * length_factor * type_match_bonus
            
            scored_sentences.append((sentence, final_score))
        
        # 3. Select and organize the most relevant sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3-5 sentences depending on scores
        top_sentences = [s for s, score in scored_sentences[:5] if score > 0.5]
        
        # 4. Construct a coherent answer
        if top_sentences:
            # If we have only 1-2 sentences, just join them
            if len(top_sentences) <= 2:
                answer = " ".join(top_sentences)
            else:
                # For 3+ sentences, create a more structured response
                if question_type in ["definition", "descriptive", "explanation"]:
                    answer = "Based on the course materials: " + " ".join(top_sentences)
                elif question_type == "enumeration":
                    # Format as a list for enumeration questions
                    answer = "From the course materials:\n- " + "\n- ".join(top_sentences)
                else:
                    answer = " ".join(top_sentences)
            
            return answer
        else:
            return "I couldn't find specific information to answer your question in the provided course materials."
    
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
        """Improved answer function with better relevance scoring and answer formatting"""
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # First, handle special cases like questions about the overall document topic
        query_lower = query.lower()
        
        # Special case handling for general document questions
        if any(phrase in query_lower for phrase in ["what is this pdf about", "what is this document about", 
                "main topic", "which topic", "what topic", "subject of this", "this pdf explain"]):
            
            # For questions about the document's overall topic, take a different approach
            # Analyze the first few paragraphs which typically contain topic information
            paragraphs = context.split('\n\n')
            intro_text = '\n\n'.join(paragraphs[:min(3, len(paragraphs))])
            
            # Extract likely topic indicators from the introduction
            topic_indicators = ["introduction", "overview", "about", "this paper", "this document", 
                              "we present", "discusses", "examines", "explores", "focuses on"]
            
            topic_sentences = []
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', intro_text)
            
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in topic_indicators):
                    topic_sentences.append(sentence)
            
            # If we found specific topic sentences, use them
            if topic_sentences:
                result = "This document appears to be about " + " ".join(topic_sentences[:2])
                return {"result": result, "source_documents": docs}
            
            # Otherwise, use the first 2-3 sentences which often state the document's purpose
            else:
                result = "Based on the document content, it appears to cover: " + " ".join(sentences[:3])
                return {"result": result, "source_documents": docs}
                
        # Regular question handling continues
        # Identify question type
        question_starters = {
            "what": "descriptive",
            "who": "entity",
            "when": "temporal",
            "where": "location",
            "why": "reasoning",
            "how": "process",
            "which": "selection",
            "can": "possibility",
            "does": "verification",
            "is": "verification",
            "are": "verification",
            "do": "verification",
            "define": "definition",
            "explain": "explanation",
            "compare": "comparison",
            "contrast": "comparison",
            "list": "enumeration",
            "describe": "descriptive"
        }
        
        # Determine question type
        question_type = "general"
        for starter, q_type in question_starters.items():
            if query_lower.startswith(starter) or f" {starter} " in query_lower:
                question_type = q_type
                break
                
        # Extract meaningful keywords (excluding common words)
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'to', 'and', 'or', 
                       'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'this', 'that', 
                       'these', 'those', 'it', 'its', 'they', 'them', 'their', 'he', 'she', 'him', 'her', 
                       'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose'}
        
        # Extract all words
        all_words = re.findall(r'\b\w+\b', query_lower)
        
        # Filter important keywords
        keywords = [k for k in all_words if k not in common_words and len(k) > 2]
        
        # If we have too few keywords, include some common question words that might be important
        if len(keywords) < 2:
            question_specific_words = ['what', 'when', 'where', 'why', 'how', 'which', 'who']
            for word in all_words:
                if word in question_specific_words and word not in keywords:
                    keywords.append(word)
        
        # 2. Score sentences based on multiple factors
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence.split()) < 5:
                continue
                
            sentence_lower = sentence.lower()
            
            # Multiple scoring factors:
            # a. Keyword matches (basic relevance)
            # b. Keyword density (prefer sentences with higher concentration of keywords)
            # c. Sentence length (prefer reasonable length sentences)
            # d. Question type matching (e.g., "when" questions should match sentences with dates)
            
            # Basic keyword match count
            keyword_count = sum(1 for k in keywords if k in sentence_lower)
            
            # Skip sentences with no keyword matches
            if keyword_count == 0:
                continue
                
            # Calculate keyword density (matches per word)
            word_count = len(sentence.split())
            keyword_density = keyword_count / word_count if word_count > 0 else 0
            
            # Length factor (prefer medium-length sentences)
            length_factor = 1.0
            if 10 <= word_count <= 25:  # Ideal length
                length_factor = 1.5
            elif word_count > 40:  # Too long
                length_factor = 0.7
                
            # Question type matching bonus
            type_match_bonus = 1.0
            if question_type == "temporal" and re.search(r'\b(in|on|during|year|date|when|time|century|decade|period|era|age)\b', sentence_lower):
                type_match_bonus = 2.0
            elif question_type == "location" and re.search(r'\b(in|at|on|near|location|place|where|region|area|country|city|state)\b', sentence_lower):
                type_match_bonus = 2.0
            elif question_type == "entity" and re.search(r'\b(person|people|who|name|individual|group|organization|company)\b', sentence_lower):
                type_match_bonus = 2.0
            elif question_type == "reasoning" and re.search(r'\b(because|since|reason|cause|effect|result|due to|why|therefore)\b', sentence_lower):
                type_match_bonus = 2.0
            elif question_type == "definition" and re.search(r'\b(is|are|refers to|defined as|means|definition)\b', sentence_lower):
                type_match_bonus = 2.0
            
            # Calculate final score
            final_score = keyword_count * keyword_density * length_factor * type_match_bonus
            
            scored_sentences.append((sentence, final_score))
        
        # 3. Select and organize the most relevant sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3-5 sentences depending on scores
        top_sentences = [s for s, score in scored_sentences[:5] if score > 0.5]
        
        # 4. Construct a coherent answer
        if top_sentences:
            # If we have only 1-2 sentences, just join them
            if len(top_sentences) <= 2:
                answer = " ".join(top_sentences)
            else:
                # For 3+ sentences, create a more structured response
                if question_type in ["definition", "descriptive", "explanation"]:
                    answer = "Based on the course materials: " + " ".join(top_sentences)
                elif question_type == "enumeration":
                    # Format as a list for enumeration questions
                    answer = "From the course materials:\n- " + "\n- ".join(top_sentences)
                else:
                    answer = " ".join(top_sentences)
                    
            # Ensure answer is not too long
            if len(answer) > 600:
                answer = answer[:597] + "..."
            
            return {"result": answer, "source_documents": docs}
        else:
            return {"result": "I couldn't find specific information to answer your question in the provided course materials.", "source_documents": docs}
    
    # Return a callable object that mimics the chain interface
    class SimpleQAChain:
        def __init__(self, answer_func):
            self.answer_func = answer_func
            
        def __call__(self, query):
            return self.answer_func(query)
    
    return SimpleQAChain(answer_function)
