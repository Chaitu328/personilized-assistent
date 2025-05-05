import os
import json
import re
from utils import split_text_into_chunks, clean_text

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
        # Simple rule-based flashcard generation
        # Extract sentences that look like they contain facts, definitions or key concepts
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        potential_cards = []
        
        # Extract potential definitions
        definition_patterns = [
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+(?:is|are|refers to|means|describes)\s+([^\.]+)',
            r'([A-Z][a-z]+(?:\s+[a-z]+)*):?\s+([^\.]+)'
        ]
        
        # Look for sentences that match definition patterns
        for sentence in sentences:
            for pattern in definition_patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    if len(match) >= 2 and len(match[0]) > 3 and len(match[1]) > 10:
                        term = match[0].strip()
                        definition = match[1].strip()
                        potential_cards.append({
                            "question": f"What is {term}?",
                            "answer": definition
                        })
        
        # If we don't have enough cards from definitions, add factual questions
        if len(potential_cards) < num_cards:
            fact_patterns = [
                r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+([a-z]+(?:\s+[a-z]+)*)\s+([^\.]+)'  # Subject verb object
            ]
            
            for sentence in sentences:
                if len(potential_cards) >= num_cards:
                    break
                    
                # Skip very short or very long sentences
                if len(sentence.split()) < 5 or len(sentence.split()) > 25:
                    continue
                    
                # Create a question by converting a statement to a question
                # For example: "The Earth orbits the Sun" -> "What orbits the Sun?"
                words = sentence.split()
                if len(words) >= 5:
                    subject = ' '.join(words[:2])
                    remainder = ' '.join(words[2:])
                    potential_cards.append({
                        "question": f"What {remainder}?",
                        "answer": subject + " " + remainder
                    })
                    
        # Ensure we only return the requested number of unique cards
        unique_cards = []
        questions_seen = set()
        
        for card in potential_cards:
            q = card["question"].lower()
            if q not in questions_seen and len(unique_cards) < num_cards:
                questions_seen.add(q)
                unique_cards.append(card)
                
        return unique_cards[:num_cards]
    
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
