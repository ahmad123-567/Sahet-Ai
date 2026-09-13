import os
import glob
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
KNOWLEDGE_DIR = "knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================
# RAG PIPELINE: PDF LOADING & VECTOR STORE
# ==========================================

@st.cache_resource(show_spinner=False)
def initialize_knowledge_base():
    """
    Loads all PDFs from knowledge/, extracts text, splits into chunks,
    generates HuggingFace embeddings, and builds a cached FAISS vector database.
    """
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        return None, 0

    pdf_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.pdf"))
    if not pdf_files:
        return None, 0

    documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
        except Exception as e:
            st.warning(f"Failed to process PDF {pdf_path}: {str(e)}")

    if not documents:
        return None, 0

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Generate Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store, len(chunks)


# ==========================================
# AGENT IMPLEMENTATIONS
# ==========================================

def safety_assessment_agent(symptoms, question):
    """
    Agent 3: Evaluates symptoms and question text for critical medical red flags.
    """
    urgent_keywords = [
        "unconscious", "fainted", "breathing difficulty", "severe pain", 
        "gasping", "bluish lips", "high fever", "convulsions", "seizure", 
        "severe dehydration", "bleeding", "blood in stool", "blood in vomit"
    ]
    
    text_to_check = (question + " " + " ".join(symptoms)).lower()
    is_emergency = any(kw in text_to_check for kw in urgent_keywords)
    
    reason = "Critical emergency warning signs detected in user query or selected symptoms." if is_emergency else "No critical immediate warning flags detected."
    return {"is_emergency": is_emergency, "reason": reason}


def rag_knowledge_agent(vector_store, query, situation_type, k=4):
    """
    Agent 2: Retrieves relevant context chunks from FAISS vector store.
    """
    if vector_store is None:
        return []
    
    augmented_query = f"{situation_type}: {query}"
    retrieved_docs = vector_store.similarity_search(augmented_query, k=k)
    return retrieved_docs


def call_groq_llm(api_key, model_name, system_prompt, user_prompt):
    """
    Helper function to query Groq LLM safely with error handling.
    """
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Low temperature for accurate, grounded facts
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API Error: {str(e)}. Please check your API key and configuration."


def health_guidance_agent(api_key, model_name, situation_analysis, context_text, question, language):
    """
    Agent 4: Synthesizes health guidance grounded strictly in retrieved RAG context.
    """
    system_prompt = """
You are SehatAI Pakistan, an expert public health and disaster response AI assistant.
Your job is to provide accurate, helpful, clear, and grounded public health guidance during disasters in Pakistan.

STRICT RAG RULES:
1. Ground your response primarily in the provided RETRIEVED CONTEXT.
2. If the answer cannot be determined from the retrieved context, state clearly: "I could not find enough verified information in my knowledge base to answer this confidently."
3. Do NOT hallucinate medical facts, drug dosages, specific hospital names, or emergency numbers.
4. Do NOT diagnose diseases or prescribe medication.
5. Structure your output clearly using markdown headings.

REQUIRED RESPONSE STRUCTURE:
🧭 **Situation Summary**
⚠️ **Health Risk Awareness**
🟢 **What You Can Do Now**
🛡️ **Prevention Tips**
🚨 **Warning Signs**
🏥 **When to Seek Medical Help**

LANGUAGE RULE:
You MUST generate the entire output in the user's requested language choice: English, Urdu, or Roman Urdu.
"""

    user_prompt = f"""
USER CONTEXT:
- Disaster/Situation: {situation_analysis['disaster_type']}
- Location: {situation_analysis['location']}
- Age Group: {situation_analysis['age_group']}
- Symptoms: {situation_analysis['symptoms']}
- Requested Language: {language}

RETRIEVED KNOWLEDGE CONTEXT:
{context_text if context_text else "No relevant context found in vector database."}

USER QUESTION:
{question}

Generate structured guidance according to the required sections in {language}.
"""

    return call_groq_llm(api_key, model_name, system_prompt, user_prompt)


def preparedness_agent(api_key, model_name, situation_type, language):
    """
    Agent 5: Generates a disaster preparedness checklist based on scenario.
    """
    system_prompt = "You are a disaster preparedness expert for Pakistan. Output a clean, bulleted checklist with markdown checkboxes ([ ]) for practical disaster health preparedness."
    
    user_prompt = f"""
Generate a practical disaster preparedness checklist for the scenario: {situation_type}.
Language: {language}.
Keep items practical for households in Pakistan. Maximum 6 bullet points.
"""

    return call_groq_llm(api_key, model_name, system_prompt, user_prompt)
