import os
import re
from collections import Counter
from utils import extract_topics, split_text_into_chunks

# Define common stopwords (for text analysis)
STOPWORDS = {
    'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these', 'those',
    'then', 'just', 'so', 'than', 'such', 'when', 'who', 'how', 'where', 'why', 'is', 'are', 'am', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'to', 'from', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'you', 'your'
}

def identify_topics(text, retriever):
    """
    Identify main topics from the text
    
    Args:
        text (str): The text to extract topics from
        retriever: The retriever to use for RAG
        
    Returns:
        list: List of identified topics
    """
    system_prompt = (
        "You are a knowledgeable teaching assistant that helps identify the main topics "
        "in educational materials. Extract the key topics that would be useful for "
        "creating a topic-wise summary of the content."
    )
    
    user_prompt = (
        "Analyze the following course material and identify the main topics covered. "
        "These topics will be used to create topic-wise summaries for students. "
        "List 5-8 main topics, with each topic being a short phrase (2-5 words). "
        "Format your response as a JSON array of strings.\n\n"
    )
    
    # Use retrieval to get a good overview of the document
    query = "What are the main topics and themes in this document?"
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    full_prompt = user_prompt + context
    
    try:
        # Simple text-based topic extraction
        # Count word frequencies and find common phrases
        
        # Clean and tokenize the text
        clean_text = context.lower()
        words = re.findall(r'\b\w+\b', clean_text)
        words = [w for w in words if w not in STOPWORDS and len(w) > 3]
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Find most common words
        common_words = [word for word, count in word_counts.most_common(20)]
        
        # Extract phrases around common words
        phrases = []
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean_text)
        
        for word in common_words:
            for sentence in sentences:
                if word in sentence:
                    # Find noun phrases containing the word
                    for match in re.finditer(r'\b([a-z]+\s+){0,2}' + word + r'(\s+[a-z]+){0,2}\b', sentence):
                        phrase = match.group(0).strip()
                        if 2 <= len(phrase.split()) <= 5:  # Ensure phrase is 2-5 words
                            phrases.append(phrase)
        
        # Get unique phrases and take top 5-8
        unique_phrases = list(set(phrases))
        topics = sorted(unique_phrases, key=lambda x: sum(word_counts.get(w, 0) for w in x.split()))[-8:]
        
        # If we couldn't extract enough topics, use the top words as single-word topics
        if len(topics) < 5:
            for word in common_words:
                if len(topics) >= 8:
                    break
                if word not in [t.split()[0] for t in topics]:
                    topics.append(word)
        
        # Capitalize topic phrases
        topics = [' '.join(word.capitalize() for word in topic.split()) for topic in topics]
        
        return topics
    
    except Exception as e:
        print(f"Error identifying topics: {e}")
        # Fallback to basic topic extraction
        return extract_topics(text)

def generate_summary_for_topic(topic, retriever):
    """
    Generate a summary for a specific topic using RAG
    
    Args:
        topic (str): The topic to summarize
        retriever: The retriever to use for RAG
        
    Returns:
        str: Summary of the topic
    """
    system_prompt = (
        "You are an educational assistant that creates clear, informative summaries for students. "
        "Create a concise but comprehensive summary of the provided course material focusing on "
        "the specified topic. Include key concepts, definitions, and relationships."
    )
    
    user_prompt = f"Create a summary focusing on the topic: {topic}\n\n"
    
    # Use retrieval to get relevant sections for this topic
    retrieved_docs = retriever.get_relevant_documents(topic)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    full_prompt = user_prompt + "Here are the relevant sections from the course material:\n\n" + context
    
    try:
        # Simple text-based approach to generate a summary
        # Extract sentences that mention the topic or related terms
        
        # Break the topic into words
        topic_words = set(topic.lower().split())
        
        # Split context into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        
        # Score each sentence based on relevance to the topic
        scored_sentences = []
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue
                
            sentence_lower = sentence.lower()
            
            # Basic scoring: count how many topic words appear in the sentence
            # and give higher weight to sentences that contain the exact topic phrase
            score = 0
            for word in topic_words:
                if word in sentence_lower and word not in STOPWORDS:
                    score += 1
                    
            # Bonus for exact topic phrase
            if topic.lower() in sentence_lower:
                score += 3
                
            if score > 0:
                scored_sentences.append((sentence, score))
                
        # Sort sentences by score and take the top ones for the summary
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected_sentences = [s[0] for s in scored_sentences[:5]]
        
        # If we found enough relevant sentences, construct a summary
        if selected_sentences:
            summary = ' '.join(selected_sentences)
            return summary
        else:
            return f"No specific information about '{topic}' found in the course material."
    
    except Exception as e:
        print(f"Error generating summary for topic '{topic}': {e}")
        return f"Failed to generate summary for this topic. Error: {str(e)}"

def generate_summaries(text, retriever):
    """
    Generate topic-wise summaries from the provided text
    
    Args:
        text (str): The text to generate summaries from
        retriever: The retriever to use for RAG
        
    Returns:
        dict: Dictionary mapping topics to their summaries
    """
    # First, identify the main topics
    topics = identify_topics(text, retriever)
    
    # Generate a summary for each topic
    summaries = {}
    for topic in topics:
        if isinstance(topic, str) and topic.strip():  # Ensure topic is a non-empty string
            summary = generate_summary_for_topic(topic, retriever)
            summaries[topic] = summary
    
    return summaries
