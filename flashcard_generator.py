import os
import json
from groq import Groq
from utils import split_text_into_chunks, clean_text

# Initialize Groq client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

def generate_flashcards(text, retriever, num_cards=10):
    """
    Generate flashcards from the provided text using OpenAI API
    
    Args:
        text (str): The text to generate flashcards from
        retriever: The retriever to use for RAG
        num_cards (int): Number of flashcards to generate
        
    Returns:
        list: List of flashcards, each containing a question and answer
    """
    # Generate initial flashcards based on the entire text
    system_prompt = (
        "You are an educational assistant that creates high-quality flashcards for students. "
        "Generate flashcards that test important concepts, definitions, and relationships from the course material. "
        "Focus on key information that would be valuable for a student to learn. "
        "Each flashcard should have a clear question and a concise, correct answer."
    )
    
    user_prompt = (
        f"Generate {num_cards} flashcards from the following course material. "
        f"Identify the most important concepts, facts, definitions, and relationships. "
        f"Format your response as a JSON array of objects, each with 'question' and 'answer' fields.\n\n"
    )
    
    # Use retrieval for generating the flashcards
    query = "What are the key concepts, definitions, and facts from this course material?"
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
        
        # Parse the response
        cards_json = json.loads(response.choices[0].message.content)
        
        # Ensure we have a list of flashcards
        if "flashcards" in cards_json:
            flashcards = cards_json["flashcards"]
        else:
            # If the model didn't use the exact "flashcards" key, just use the first array found
            for key, value in cards_json.items():
                if isinstance(value, list) and len(value) > 0:
                    flashcards = value
                    break
            else:
                # Fallback if no array is found
                flashcards = []
        
        return flashcards[:num_cards]  # Ensure we only return the requested number
    
    except Exception as e:
        print(f"Error generating flashcards: {e}")
        # Return a simple error flashcard
        return [{"question": "Error", "answer": f"Failed to generate flashcards: {str(e)}"}]

def validate_flashcards(flashcards, text):
    """
    Validate the generated flashcards against the original text
    
    Args:
        flashcards (list): The flashcards to validate
        text (str): The original text
        
    Returns:
        list: Validated and possibly corrected flashcards
    """
    validated_cards = []
    
    for card in flashcards:
        question = card.get("question", "")
        answer = card.get("answer", "")
        
        # Simple validation: check if key terms from the answer appear in the text
        answer_words = set(answer.lower().split())
        text_lower = text.lower()
        
        # Remove common words for more meaningful validation
        common_words = {"the", "a", "an", "in", "on", "at", "of", "for", "with", "by", "to", "and", "or", "but"}
        key_words = answer_words - common_words
        
        valid = True
        for word in key_words:
            if len(word) > 3 and word not in text_lower:  # Only check reasonably long words
                valid = False
                break
        
        if valid:
            validated_cards.append(card)
    
    return validated_cards
