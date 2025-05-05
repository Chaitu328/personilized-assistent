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
        
        # Display the flashcards in a more visually appealing format
        st.write("### Study Flashcards")
        st.write("These flashcards were automatically generated from your course material. Click on each card to reveal the answer.")
        
        # Create columns for better layout
        cols = st.columns(2)
        
        for i, card in enumerate(st.session_state.flashcards):
            # Alternate between columns for a nicer grid layout
            col_idx = i % 2
            
            # Format each flashcard as a clean card with a reveal effect
            with cols[col_idx]:
                card_container = st.container()
                with card_container:
                    # Apply styling to make it look like a card
                    st.markdown(f"""
                    <div style="border:1px solid #cccccc; border-radius:10px; padding:15px; margin-bottom:15px; background-color:#f8f9fa;">
                        <h4 style="margin-top:0px; color:#1f77b4;">Flashcard {i+1}</h4>
                        <p><strong>Question:</strong> {card['question']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a toggle for the answer
                    show_answer = st.checkbox(f"Show Answer #{i+1}", key=f"answer_{i}")
                    
                    if show_answer:
                        st.markdown(f"""
                        <div style="border:1px solid #dddddd; border-radius:10px; padding:15px; margin-bottom:20px; background-color:#e8f4f8;">
                            <p><strong>Answer:</strong> {card['answer']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("<hr style='margin: 15px 0px; opacity: 0.3;'>", unsafe_allow_html=True)
        
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
        
        # Display the summaries with better styling
        st.write("### Topic Summaries")
        st.write("These summaries highlight the key concepts from your course material. Click on each topic to expand.")
        
        # Create a card-based layout for summaries
        for topic, summary in st.session_state.summaries.items():
            # Create an attractive topic header
            st.markdown(f"""
            <div style="background-color:#f0f8ff; padding:10px; border-radius:10px; margin-bottom:12px; border-left: 5px solid #007bff;">
                <h4 style="margin: 0px; color:#007bff;">{topic}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Read Summary"):
                # Format the summary text
                st.markdown(f"""
                <div style="background-color:#f9f9f9; padding:15px; border-radius:8px; border:1px solid #eaeaea;">
                    {summary}
                </div>
                """, unsafe_allow_html=True)
                
            # Add whitespace between topics
            st.write("")
        
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
        # Create a more engaging Q&A interface
        st.markdown("""
        <div style="background-color:#f5f5f5; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h4 style="margin-top:0;">Ask me anything about the course material</h4>
            <p>Examples: "What are the key concepts in this course?", "Can you explain how [topic] works?", "What is the definition of [term]?"</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Store conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
            
        # Input for question with a button for better UX
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input("Enter your question:", placeholder="Type your question here...")
        with col2:
            search_button = st.button("Ask", type="primary")
            
        # Process question when button is clicked or Enter is pressed
        if (question and search_button) or (question and question != st.session_state.get('last_question', '')):
            st.session_state.last_question = question
            
            with st.spinner("Finding the answer..."):
                retriever = get_retriever(st.session_state.vector_store)
                answer = answer_question(question, retriever)
                
                # Add to conversation history
                st.session_state.conversation_history.append({"question": question, "answer": answer})
            
            # Clear the input field after submission
            # (This doesn't work directly in Streamlit, but keeps the code ready for when the feature is available)
                
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("### Conversation")
            
            for i, exchange in enumerate(st.session_state.conversation_history):
                # Format the question
                st.markdown(f"""
                <div style="background-color:#e9f7fe; padding:10px; border-radius:10px; margin-bottom:10px; border-left:4px solid #4285f4;">
                    <p style="margin:0px; font-weight:bold;">You asked:</p>
                    <p style="margin:0px;">{exchange["question"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Format the answer
                st.markdown(f"""
                <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px; border-left:4px solid #34a853;">
                    <p style="margin:0px; font-weight:bold;">Answer:</p>
                    <p style="margin-top:5px;">{exchange["answer"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add source documents in an expander
                with st.expander("View source material"):
                    docs = retriever.get_relevant_documents(exchange["question"])
                    for j, doc in enumerate(docs):
                        st.markdown(f"""
                        <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom:10px;">
                            <p style="margin:0; font-size:0.9em; color:#555;"><strong>Source {j+1}:</strong></p>
                            <p style="margin:5px 0 0 0;">{doc.page_content}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Add option to clear conversation history
            if st.button("Clear Conversation"):
                st.session_state.conversation_history = []
                st.rerun()

# Add a footer
st.markdown("---")
st.markdown("**Personalized Learning Assistant** • Created using Streamlit and RAG Technology")
