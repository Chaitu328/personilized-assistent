import os
from groq import Groq
from utils import extract_topics, split_text_into_chunks

# Initialize Groq client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

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
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",  # Using Llama 3 70B model from Groq
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
        )
        
        # Parse the response to get the topics
        import json
        result = json.loads(response.choices[0].message.content)
        
        # Handle different possible response formats
        if isinstance(result, dict) and "topics" in result:
            topics = result["topics"]
        elif isinstance(result, dict) and any(isinstance(result.get(k), list) for k in result):
            # Find the first list in the response
            for k, v in result.items():
                if isinstance(v, list):
                    topics = v
                    break
        elif isinstance(result, list):
            topics = result
        else:
            # Fallback to basic topic extraction if the API response isn't as expected
            topics = extract_topics(text)
        
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
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",  # Using Llama 3 70B model from Groq
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=800,
        )
        
        return response.choices[0].message.content
    
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
