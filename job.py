# =============================================================================
# SECTION 1: Environment Setup and Imports
# =============================================================================
import os
# Disable Streamlit file watching for problematic modules
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"

# Add this at the very top of your file, before any other imports
import streamlit as st

import re
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import spacy
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# =============================================================================
# SECTION 2: Session State Initialization
# =============================================================================
# Initialize session state for models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'faiss_store' not in st.session_state:
    st.session_state.faiss_store = None
if 'nlp' not in st.session_state:
    st.session_state.nlp = None

# =============================================================================
# SECTION 3: Model Loading Functions
# =============================================================================
# Add spaCy model loading
def load_spacy_model(model_name="en_core_web_sm"):
    print(f"[load_spacy_model] Loading spaCy model: {model_name}")
    try:
        nlp = spacy.load(model_name)
        print("[load_spacy_model] spaCy model loaded successfully.")
        return nlp
    except Exception as e:
        print(f"[load_spacy_model] Error loading spaCy model: {e}")
        st.error(f"Could not load spaCy model '{model_name}'. Please ensure it is installed (`python -m spacy download {model_name}`).")
        return None

# =============================================================================
# SECTION 4: PDF Processing Functions
# =============================================================================
# 1. PDF Text Extraction
@st.cache_data
def extract_text(pdf_path):
    print(f"[extract_text] Processing: {pdf_path}")
    doc = fitz.open(pdf_path)
    text_by_page = []
    total_pages = len(doc)
    
    for page_num, page in enumerate(doc):
        print(f"[extract_text] Page {page_num+1}/{total_pages}")
        text = page.get_text()
        text_by_page.append({
            "text": text,
            "page": page_num,
            "pdf": os.path.basename(pdf_path)
        })
    
    print(f"[extract_text] Finished: {pdf_path}")
    return text_by_page

# 2. Chunking & Embedding
@st.cache_data
def split_texts(_text_by_page, chunk_size=500, chunk_overlap=20):
    print(f"[split_texts] Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for page_data in _text_by_page:
        page_chunks = splitter.split_text(page_data["text"])
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page_data["page"],
                "pdf": page_data["pdf"]
            })
    print(f"[split_texts] Total chunks: {len(chunks)}")
    return chunks

# =============================================================================
# SECTION 5: FAISS Index Functions
# =============================================================================
# 3. FAISS Index
@st.cache_resource
def build_or_load_faiss(chunks, _embeddings, index_path="faiss_index"):
    print(f"[build_or_load_faiss] Building or loading FAISS index...")
    version = "v1-chunk500-miniLM"  # Current version identifier
    version_file = os.path.join(index_path, "version.txt")
    
    if os.path.exists(index_path) and os.path.exists(version_file):
        with open(version_file, 'r') as f:
            stored_version = f.read().strip()
        if stored_version == version:
            print(f"[build_or_load_faiss] Loading existing index from {index_path}")
            return FAISS.load_local(index_path, _embeddings, allow_dangerous_deserialization=True)
    
    # If version mismatch or index doesn't exist, create new index
    print(f"[build_or_load_faiss] Creating new index...")
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"page": chunk["page"], "pdf": chunk["pdf"]} for chunk in chunks]
    faiss_store = FAISS.from_texts(texts, _embeddings, metadatas=metadatas)
    
    # Save index and version
    os.makedirs(index_path, exist_ok=True)
    faiss_store.save_local(index_path)
    with open(version_file, 'w') as f:
        f.write(version)
    print(f"[build_or_load_faiss] Index built and saved to {index_path}")
    return faiss_store

# =============================================================================
# SECTION 6: Model Setup
# =============================================================================
# 4. Model Setup
@st.cache_resource
def load_models():
    print("[load_models] Loading models...")
    try:
        # Force CPU for all models
        torch.set_num_threads(4)  # Limit CPU threads
        
        # Load the embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Load the cross-encoder for re-ranking
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        
        # Load GPT-Neo model and tokenizer
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        
        print("[load_models] Models loaded successfully")
        return embedding_model, cross_encoder, model, tokenizer
    except Exception as e:
        print(f"[load_models] Error loading models: {e}")
        raise e

# =============================================================================
# SECTION 7: Answer Generation Functions
# =============================================================================
# 6. Answer Generation
def generate_response(prompt: str, model, tokenizer) -> str:
    print(f"[generate_response] Generating response for prompt of length {len(prompt)}")
    
    try:
        # Generate the text with parameters optimized for detailed answers
        gen_tokens = model.generate(
            tokenizer(prompt, return_tensors="pt").input_ids,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_length=1024,
            max_new_tokens=300,
            num_beams=5,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        # Decode the generated tokens
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        
        # Extract just the answer part
        if "Answer:" in gen_text:
            answer = gen_text.split("Answer:")[-1].strip()
        else:
            answer = gen_text.strip()
        
        # Process the answer to ensure complete sentences and remove questions
        sentences = answer.split('.')
        complete_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Only process non-empty sentences
                # Skip sentences that end with question marks
                if sentence.endswith('?'):
                    continue
                # Check if the sentence is complete (has a subject and verb)
                if len(sentence.split()) >= 3:  # Basic check for sentence completeness
                    complete_sentences.append(sentence)
        
        # Join the complete sentences
        if complete_sentences:
            answer = '. '.join(complete_sentences) + '.'
        else:
            answer = "I don't know."
        
        print(f"[generate_response] Generated answer: {answer}")
        return answer
    except Exception as e:
        print(f"[generate_response] Error generating response: {e}")
        return "I don't know."

# 8. Question Answering Pipeline
def answer_query(query, faiss_store, embedding_model, cross_encoder, model, tokenizer, nlp, k=5, distance_threshold=0.9):
    print(f"\n[answer_query] Starting query processing for: {query}")
    try:
        # Perform the search and get scores
        docs_and_scores = faiss_store.similarity_search_with_score(query, k=k)

        if not docs_and_scores:
            print("[answer_query] No documents found from FAISS.")
            return "I'm sorry, but I couldn't find any relevant information in the documents for your query.", []

        print(f"[answer_query] Retrieved {len(docs_and_scores)} documents with scores from FAISS.")
        for i, (doc, score) in enumerate(docs_and_scores):
            page_num = doc.metadata.get('page', 'N/A')
            pdf_name = doc.metadata.get('pdf', 'N/A')
            print(f"  Doc {i}: score={score:.4f}, page={page_num}, pdf={pdf_name}")

        # Check if the most relevant document's score is above the threshold
        best_score = docs_and_scores[0][1]
        if best_score > distance_threshold:
            print(f"[answer_query] Best score {best_score:.4f} is above threshold {distance_threshold}. Query likely off-topic.")
            return "I can only answer questions based on the provided anatomy PDF documents. Your question seems unrelated to their content.", []

        # Use documents that were initially retrieved if the best one passed the initial check
        docs = [doc for doc, score in docs_and_scores]

        # Extract the relevant context from the documents
        context = " ".join([doc.page_content for doc in docs])
        
        # Determine if we have enough context for a detailed answer
        context_length = len(context.split())
        is_detailed = context_length > 200  # Adjust this threshold as needed

        # Construct the prompt based on available context
        if is_detailed:
            prompt = (
                "You are a helpful assistant answering questions based ONLY on the provided context from anatomy PDFs. "
                "Provide a detailed and comprehensive answer that includes:\n"
                "1. A clear definition or description\n"
                "2. Key anatomical features and structures\n"
                "3. Important functions or roles\n"
                "4. Relevant relationships with other structures\n"
                "Use proper medical terminology and ensure all sentences are complete.\n"
                "Do not include any questions in your answer.\n"
                "If the answer cannot be found within the given context, respond with 'I don't know.'\n\n"
                f"Context: {context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )
        else:
            prompt = (
                "You are a helpful assistant answering questions based ONLY on the provided context from anatomy PDFs. "
                "Provide a clear and concise answer using proper medical terminology.\n"
                "Do not include any questions in your answer.\n"
                "If the answer cannot be found within the given context, respond with 'I don't know.'\n\n"
                f"Context: {context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )

        # Generate the response
        response = generate_response(prompt, model, tokenizer)

        return response, docs
    except Exception as e:
        print(f"[answer_query] Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return "An error occurred while trying to answer your question. Please try again.", []

# =============================================================================
# SECTION 8: Streamlit UI and Main Execution
# =============================================================================
# 9. Streamlit UI
def main():
    print("[main] Starting Streamlit app...")
    st.title("Book Q&A Bot")
    data_dir = "data"

    # Initialize models and data if not already loaded
    if not st.session_state.models_loaded:
        with st.spinner("Loading models and data..."):
            # Load spaCy model
            st.session_state.nlp = load_spacy_model()
            if st.session_state.nlp is None:
                return

            # Load PDFs, chunk, embed, and build FAISS index
            print("[main] Loading and processing PDFs...")
            text_by_page = []
            if not os.path.exists(data_dir) or not os.listdir(data_dir):
                st.error(f"Data directory '{data_dir}' is empty or does not exist. Please add your PDF files there.")
                return

            for pdf_file in os.listdir(data_dir):
                if pdf_file.endswith(".pdf"):
                    pdf_path = os.path.join(data_dir, pdf_file)
                    text_by_page.extend(extract_text(pdf_path))

            if not text_by_page:
                st.error("No text could be extracted from PDFs in the data directory.")
                return

            chunks = split_texts(text_by_page)
            if not chunks:
                st.error("No chunks could be created from the extracted text.")
                return

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.faiss_store = build_or_load_faiss(chunks, embeddings)
            
            # Load models
            print("[main] Loading models...")
            try:
                st.session_state.embedding_model, st.session_state.cross_encoder, st.session_state.model, st.session_state.tokenizer = load_models()
            except Exception as e:
                st.error(f"Failed to load transformer models: {e}")
                return
            
            st.session_state.models_loaded = True
            st.success("Models and data loaded successfully!")

    # UI
    user_query = st.text_input("Ask a question about the book:")
    if user_query:
        print(f"[main] User query: {user_query}")
        if not st.session_state.models_loaded or st.session_state.faiss_store is None:
            st.error("Models or FAISS store not loaded. Please ensure PDFs are processed and models are loaded.")
            return

        with st.spinner("Generating answer..."):
            answer, _ = answer_query(
                user_query, 
                st.session_state.faiss_store,
                st.session_state.embedding_model,
                st.session_state.cross_encoder,
                st.session_state.model,
                st.session_state.tokenizer,
                st.session_state.nlp,
                distance_threshold=0.9
            )
            
            # Display only the answer
            st.write("### Answer")
            st.write(answer)

if __name__ == "__main__":
    print("[__main__] Running main()")
    main()



