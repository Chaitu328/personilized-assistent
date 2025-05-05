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
        # Improved flashcard generation that focuses on clear questions and concise answers
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        potential_cards = []
        
        # Look for definition sentences - these make good flashcards
        for sentence in sentences:
            # Skip very short or very long sentences
            if len(sentence.split()) < 5 or len(sentence.split()) > 30:
                continue
                
            sentence = sentence.strip()
            
            # Pattern 1: Explicit definitions with "is defined as", "is", "refers to", etc.
            definition_match = re.search(r'([A-Z][a-zA-Z\s]+)\s+(is|are|refers to|means|is defined as|can be defined as)\s+([^\.]+)', sentence)
            if definition_match:
                term = definition_match.group(1).strip()
                definition = definition_match.group(3).strip()
                
                # Only use if term and definition are reasonable lengths
                if 2 <= len(term.split()) <= 5 and len(definition) >= 10:
                    potential_cards.append({
                        "question": f"What is {term}?",
                        "answer": definition
                    })
                    continue
            
            # Pattern 2: Key concepts with colon
            colon_match = re.search(r'([A-Z][a-zA-Z\s]+):\s+([^\.]+)', sentence)
            if colon_match:
                term = colon_match.group(1).strip()
                explanation = colon_match.group(2).strip()
                
                if len(term.split()) <= 5 and len(explanation) >= 10:
                    potential_cards.append({
                        "question": f"Explain {term}.",
                        "answer": explanation
                    })
                    continue
            
            # Pattern 3: Important facts with numbers/dates
            if re.search(r'\b(in|on|during)\s+\d{4}\b', sentence) or re.search(r'\b\d+\s+(percent|%)\b', sentence):
                # Create a factual question by removing key information
                words = sentence.split()
                if len(words) >= 7:
                    # Try to identify the subject of the sentence
                    subject_end = min(3, len(words) // 3)
                    subject = ' '.join(words[:subject_end])
                    remainder = ' '.join(words[subject_end:])
                    
                    # Create a question that asks about the factual information
                    potential_cards.append({
                        "question": f"What {remainder}?",
                        "answer": sentence
                    })
                    
        # If we still don't have enough cards, look for sentences with key educational terms
        if len(potential_cards) < num_cards:
            important_terms = ["key", "important", "significant", "essential", "fundamental", 
                               "critical", "vital", "primary", "main", "major", "central"]
            
            for sentence in sentences:
                if len(potential_cards) >= num_cards * 2:  # Generate extras for filtering
                    break
                    
                # Look for sentences that seem to be stating important concepts
                if any(term in sentence.lower() for term in important_terms):
                    # Create a question by removing the last part of the sentence
                    words = sentence.split()
                    if len(words) >= 8:
                        question_part = ' '.join(words[:len(words)//2])
                        potential_cards.append({
                            "question": f"Complete this statement: {question_part}...",
                            "answer": sentence
                        })
        
        # Ensure we have unique, quality cards
        unique_cards = []
        questions_seen = set()
        
        # First pass: eliminate exact duplicates and prefer shorter answers
        sorted_cards = sorted(potential_cards, key=lambda x: len(x["answer"]))
        
        for card in sorted_cards:
            q = card["question"].lower()
            if q not in questions_seen and len(unique_cards) < num_cards * 1.5:  # Get 50% more than needed for validation
                questions_seen.add(q)
                unique_cards.append(card)
        
        # Second pass: Ensure answers are concise (under 150 chars) and questions are clear
        final_cards = []
        for card in unique_cards:
            # Limit answer length for readability
            if len(card["answer"]) > 150:
                card["answer"] = card["answer"][:147] + "..."
                
            # Ensure question ends with ? or .
            if not card["question"].endswith("?") and not card["question"].endswith("."):
                card["question"] += "?"
                
            final_cards.append(card)
                
        return final_cards[:num_cards]  # Return only the requested number of cards
    
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
