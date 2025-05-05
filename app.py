import streamlit as st
import os
import tempfile
from PIL import Image

from pdf_processor import extract_text_from_pdf
from flashcard_generator import generate_flashcards
from summary_generator import generate_summaries
from qa_system import answer_question
from vector_store import create_vector_store, get_retriever

# Page configuration
st.set_page_config(
    page_title="Course PDF Learning Assistant",
    page_icon="📚",
    layout="wide",
)

# Initialize session state variables
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = None
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = []
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Upload"

def reset_session():
    """Reset all session state variables"""
    st.session_state.pdf_text = None
    st.session_state.pdf_name = None
    st.session_state.vector_store = None
    st.session_state.flashcards = []
    st.session_state.summaries = {}
    st.session_state.uploaded_files = []

# Sidebar with title and navigation
st.sidebar.title("Learning Assistant")
tab = st.sidebar.radio(
    "Navigate to:",
    ["Upload", "Flashcards", "Summaries", "Q&A"],
    index=["Upload", "Flashcards", "Summaries", "Q&A"].index(st.session_state.current_tab)
)
st.session_state.current_tab = tab

# Display relevant information in the sidebar
if st.session_state.pdf_name:
    st.sidebar.success(f"Working with: {st.session_state.pdf_name}")
    if st.sidebar.button("Process New PDF"):
        reset_session()
        st.rerun()

# Main content
if tab == "Upload":
    st.title("Upload Your Course Materials")
    st.write("Upload PDF files containing your course materials to generate flashcards, summaries, and answer questions.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)
    
    if uploaded_file is not None:
        with st.spinner("Processing your PDF..."):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
            
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            
            # Create vector store for RAG
            vector_store = create_vector_store(text)
            
            # Save in session state
            st.session_state.pdf_text = text
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.vector_store = vector_store
            st.session_state.uploaded_files.append(uploaded_file.name)
            
            # Clean up temp file
            os.unlink(pdf_path)
            
        st.success(f"Successfully processed {uploaded_file.name}!")
        st.write("You can now navigate to the Flashcards, Summaries, or Q&A tabs to use your document.")
        
        # Show preview of the extracted text
        with st.expander("Preview extracted text"):
            st.write(text[:1000] + "..." if len(text) > 1000 else text)

elif tab == "Flashcards":
    st.title("Flashcards")
    
    if st.session_state.pdf_text is None:
        st.warning("Please upload a PDF file first!")
    else:
        if not st.session_state.flashcards:
            with st.spinner("Generating flashcards..."):
                retriever = get_retriever(st.session_state.vector_store)
                flashcards = generate_flashcards(st.session_state.pdf_text, retriever)
                st.session_state.flashcards = flashcards
        
        # Display the flashcards
        st.write("These flashcards were automatically generated from your course material:")
        
        for i, card in enumerate(st.session_state.flashcards):
            with st.expander(f"Flashcard {i+1}: {card['question'][:80]}..."):
                st.write("**Question:**")
                st.write(card['question'])
                st.write("**Answer:**")
                st.write(card['answer'])
        
        # Option to regenerate flashcards
        if st.button("Regenerate Flashcards"):
            with st.spinner("Regenerating flashcards..."):
                retriever = get_retriever(st.session_state.vector_store)
                flashcards = generate_flashcards(st.session_state.pdf_text, retriever)
                st.session_state.flashcards = flashcards
                st.rerun()

elif tab == "Summaries":
    st.title("Topic Summaries")
    
    if st.session_state.pdf_text is None:
        st.warning("Please upload a PDF file first!")
    else:
        if not st.session_state.summaries:
            with st.spinner("Generating topic summaries..."):
                retriever = get_retriever(st.session_state.vector_store)
                summaries = generate_summaries(st.session_state.pdf_text, retriever)
                st.session_state.summaries = summaries
        
        # Display the summaries
        st.write("These summaries were automatically generated from your course material:")
        
        for topic, summary in st.session_state.summaries.items():
            with st.expander(f"Topic: {topic}"):
                st.write(summary)
        
        # Option to regenerate summaries
        if st.button("Regenerate Summaries"):
            with st.spinner("Regenerating summaries..."):
                retriever = get_retriever(st.session_state.vector_store)
                summaries = generate_summaries(st.session_state.pdf_text, retriever)
                st.session_state.summaries = summaries
                st.rerun()

elif tab == "Q&A":
    st.title("Ask Questions About Your Course Materials")
    
    if st.session_state.pdf_text is None:
        st.warning("Please upload a PDF file first!")
    else:
        question = st.text_input("Ask a question about your course material:")
        
        if question:
            with st.spinner("Finding the answer..."):
                retriever = get_retriever(st.session_state.vector_store)
                answer = answer_question(question, retriever)
                
                st.subheader("Answer")
                st.write(answer)
                
                with st.expander("Relevant context from your materials"):
                    docs = retriever.get_relevant_documents(question)
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Excerpt {i+1}:**")
                        st.write(doc.page_content)
                        st.markdown("---")

# Add a footer
st.markdown("---")
st.markdown("**Personalized Learning Assistant** • Created using Streamlit and RAG Technology")
