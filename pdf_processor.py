import PyPDF2
import re
from utils import clean_text

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        str: Extracted text from the PDF
    """
    extracted_text = ""
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                if text:
                    extracted_text += text + "\n\n"
    
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    
    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)
    return cleaned_text

def extract_metadata(pdf_path):
    """
    Extract metadata from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        dict: Metadata from the PDF
    """
    metadata = {}
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata_dict = reader.metadata
            
            if metadata_dict:
                for key, value in metadata_dict.items():
                    # Clean up the key name
                    clean_key = key.replace('/', '')
                    metadata[clean_key] = value
    
    except Exception as e:
        print(f"Error extracting metadata from PDF: {e}")
    
    return metadata

def identify_structure(text):
    """
    Identify the structure of the document based on headings, etc.
    
    Args:
        text (str): The extracted text from the PDF
    
    Returns:
        dict: Identified structure elements
    """
    structure = {
        'sections': [],
        'potential_chapters': []
    }
    
    # Look for numbered headings (e.g., "1. Introduction", "1.1 Background")
    numbered_headings = re.findall(r'\n\s*(\d+\.[\d\.]*\s+[A-Z][^\n]+)', text)
    
    # Look for capitalized headings
    capitalized_headings = re.findall(r'\n\s*([A-Z][A-Z\s]+[A-Z])\s*\n', text)
    
    # Combine all identified headings
    all_headings = numbered_headings + capitalized_headings
    
    if all_headings:
        structure['sections'] = all_headings
    
    # Identify potential chapters
    chapter_patterns = [
        r'\bCHAPTER\s+\d+',
        r'\bSection\s+\d+',
        r'\bPart\s+\d+'
    ]
    
    for pattern in chapter_patterns:
        chapters = re.findall(pattern, text, re.IGNORECASE)
        if chapters:
            structure['potential_chapters'].extend(chapters)
    
    return structure
