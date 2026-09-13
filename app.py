import os
import streamlit as st
import rag_utils as rag

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SehatAI Pakistan 🇵🇰 - Disaster Health Assistant",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design and mobile responsiveness
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E40AF;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .emergency-card {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    .sources-card {
        background-color: #EFF6FF;
        border-left: 6px solid #3B82F6;
        padding: 12px;
        border-radius: 6px;
        margin-top: 15px;
    }
    .stButton>button {
        width: 100%;
        background-color: #059669;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #047857;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# GROQ API KEY CHECK & CONFIGURATION
# ==========================================
# Fetch API key securely from Streamlit Secrets or environment variable
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

# Default configurable Groq model name
MODEL_NAME = "openai/gpt-oss-120b"

# ==========================================
# AGENTIC AI WORKFLOW DEFINITION
# ==========================================

def run_agentic_workflow(situation_type, province, district, age_group, symptoms, question, urgency, language, vector_store):
    """
    Executes the multi-agent orchestration for health guidance.
    """
    # Agent 1: Situation Analysis Agent
    st.toast("🧠 Agent 1: Analyzing user situation and location context...", icon="🔍")
    situation_analysis = {
        "disaster_type": situation_type,
        "location": f"{district}, {province}, Pakistan",
        "age_group": age_group if age_group else "Not specified",
        "symptoms": ", ".join(symptoms) if symptoms else "None reported",
        "urgency_level": urgency,
        "language": language
    }

    # Agent 3: Safety Assessment Agent (Early check)
    st.toast("🛡️ Agent 3: Checking for critical health emergency signs...", icon="🚨")
    safety_flags = rag.safety_assessment_agent(symptoms, question)

    # Agent 2: RAG Knowledge Agent
    st.toast("📚 Agent 2: Searching vector store for Pakistan-specific guidelines...", icon="📖")
    retrieved_docs = rag.rag_knowledge_agent(vector_store, question, situation_type)
    
    # Extract source filenames for transparency
    sources_used = list(set([os.path.basename(doc.metadata.get('source', 'Unknown Source')) for doc in retrieved_docs]))
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Agent 4: Health Guidance Agent
    st.toast("🩺 Agent 4: Synthesizing grounded health guidance...", icon="✍️")
    structured_response = rag.health_guidance_agent(
        GROQ_API_KEY, MODEL_NAME, situation_analysis, context_text, question, language
    )

    # Agent 5: Prevention & Preparedness Agent
    st.toast("🎒 Agent 5: Generating tailored disaster preparedness checklist...", icon="📋")
    checklist = rag.preparedness_agent(GROQ_API_KEY, MODEL_NAME, situation_type, language)

    return {
        "safety_flags": safety_flags,
        "response": structured_response,
        "checklist": checklist,
        "sources": sources_used
    }

# ==========================================
# SIDEBAR NAVIGATION & KNOWLEDGE BASE STATUS
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/32/Flag_of_Pakistan.svg", width=80)
st.sidebar.title("🇵🇰 SehatAI Pakistan")
st.sidebar.caption("Disaster Health Information System")

navigation = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🩺 Health Assistant", "🚨 Emergency Guidance", "🎒 Preparedness", "📚 Sources", "ℹ️ About"]
)

st.sidebar.divider()

# Load Vector Store
st.sidebar.subheader("📦 Knowledge Base Status")
if not GROQ_API_KEY:
    st.sidebar.error("⚠️ GROQ_API_KEY not found in secrets!")
else:
    st.sidebar.success("🔑 API Key configured")

with st.sidebar:
    with st.spinner("Initializing FAISS Vector Store..."):
        vector_store, doc_count = rag.initialize_knowledge_base()
    if vector_store:
        st.success(f"✅ RAG Active ({doc_count} PDF Chunks Loaded)")
    else:
        st.warning("⚠️ Knowledge base empty or not loaded.")

# Medical Disclaimer in Sidebar
st.sidebar.divider()
st.sidebar.info("""
**Medical Disclaimer:**
SehatAI provides general health and disaster-preparedness information based on official sources. It does NOT diagnose diseases, prescribe medicines, or replace qualified doctors or emergency services.
""")

# ==========================================
# PAGE 1: HOME PAGE
# ==========================================
if navigation == "🏠 Home":
    st.markdown('<div class="main-header">🇵🇰 SehatAI Pakistan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Disaster Health Guidance & Preparedness using RAG & Agentic AI</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Welcome to SehatAI Pakistan
        During public health emergencies, severe weather events, or natural disasters, access to accurate, localized health guidance is critical. **SehatAI Pakistan** connects citizens with verified health protocols grounded directly in authoritative public health guidelines.
        """)

        st.subheader("Key Emergency Modes Supported")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("🌊 **Flood Safety & Safe Water**\n- Waterborne disease prevention\n- Safe drinking water disinfection")
            st.markdown("☀️ **Heatwave Protection**\n- Extreme heat exposure guidelines\n- Preventing heat stroke & dehydration")
        with m_col2:
            st.markdown("🦟 **Dengue & Malaria**\n- Mosquito breeding prevention\n- Early symptom identification")
            st.markdown("🦠 **Outbreak & Hygiene**\n- Sanitation & infection control\n- Public health guidance")

        st.divider()
        st.warning("⚡ **In a Life-Threatening Emergency?** Please contact local emergency services immediately.")

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🚀 Quick Start")
        st.write("Get verified, context-aware disaster health guidance in English, Urdu, or Roman Urdu.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 2: HEALTH ASSISTANT (CORE RAG WORKFLOW)
# ==========================================
elif navigation == "🩺 Health Assistant":
    st.markdown("## 🩺 AI Disaster Health Assistant")
    st.caption("Complete the details below to receive grounded health and disaster safety guidance.")

    if not GROQ_API_KEY:
        st.error("🚨 Configuration Error: `GROQ_API_KEY` is missing. Please configure it in `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
        st.stop()

    with st.form("health_assistant_form"):
        col1, col2 = st.columns(2)

        with col1:
            situation = st.selectbox(
                "1. Select Disaster / Situation Type *",
                ["🌊 Flood", "☀️ Heatwave", "🦟 Dengue / Malaria Risk", "🦠 Disease Outbreak", "💧 Unsafe Water", "🧼 Sanitation / Hygiene Problem", "🏠 General Disaster Health", "❓ Other"]
            )

            province = st.selectbox(
                "2. Select Province *",
                ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Gilgit-Baltistan", "Azad Jammu & Kashmir", "Islamabad Capital Territory"]
            )

            district = st.text_input("City / District *", value="Lahore", help="Do NOT enter exact home address.")

            age_group = st.selectbox(
                "3. Age Group (Optional Context)",
                ["", "Child", "Teen", "Adult", "Older Adult"]
            )

        with col2:
            symptoms = st.multiselect(
                "4. Reported Symptoms (Optional — For Safety Context Only)",
                ["Fever", "Headache", "Vomiting", "Diarrhea", "Weakness", "Dizziness", "Breathing difficulty", "Skin irritation / rash", "Dehydration concern", "No symptoms / Prevention only", "Other"]
            )

            urgency = st.radio(
                "5. Urgency Level *",
                ["🟢 General Information", "🟡 Need Guidance Soon", "🔴 Possible Emergency"],
                horizontal=True
            )

            language = st.selectbox(
                "6. Preferred Response Language *",
                ["English", "Urdu", "Roman Urdu"]
            )

        question = st.text_area(
            "7. Ask Your Health or Safety Question *",
            placeholder="e.g., How can I purify water for drinking after floodwaters entered our well? Or: What should I do if someone shows signs of severe heat stroke?",
            height=120
        )

        submit_btn = st.form_submit_button("🔍 Get Health Guidance")

    if submit_btn:
        if not question.strip():
            st.warning("⚠️ Please provide a health question before proceeding.")
        elif not vector_store:
            st.error("🚨 Vector Database is not initialized. Please ensure PDF documents are present in the `knowledge/` directory.")
        else:
            # Immediate Emergency Banner Check
            if urgency == "🔴 Possible Emergency":
                st.markdown("""
                <div class="emergency-card">
                    <h3>🚨 EMERGENCY WARNING</h3>
                    <p><strong>You indicated a possible emergency.</strong> SehatAI Pakistan cannot provide emergency medical intervention or emergency services. If you or someone around you is experiencing life-threatening symptoms, seek immediate in-person medical care or contact local emergency services.</p>
                </div>
                """, unsafe_allow_html=True)

            with st.spinner("Processing request via Multi-Agent RAG Pipeline..."):
                workflow_results = run_agentic_workflow(
                    situation, province, district, age_group, symptoms, question, urgency, language, vector_store
                )

            st.divider()

            # Safety Assessment Agent Banner Output
            if workflow_results["safety_flags"]["is_emergency"]:
                st.error(f"🚨 **Safety Assessment Warning:** {workflow_results['safety_flags']['reason']}")

            # Main AI Output
            st.markdown("### 📋 Structured Health Guidance")
            st.markdown(workflow_results["response"])

            # Preparedness Checklist Tab/Expander
            st.divider()
            with st.expander("🎒 Disaster Preparedness Recommendations", expanded=True):
                st.markdown(workflow_results["checklist"])

            # Sources Transparency
            st.markdown('<div class="sources-card">', unsafe_allow_html=True)
            st.markdown("#### 📚 Knowledge Sources Used")
            if workflow_results["sources"]:
                for src in workflow_results["sources"]:
                    st.markdown(f"- 📄 `{src}`")
            else:
                st.write("No specific documents retrieved. Answer generated from general safety fallback.")
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 3: EMERGENCY GUIDANCE
# ==========================================
elif navigation == "🚨 Emergency Guidance":
    st.markdown("## 🚨 Emergency Medical Warning Signs")
    st.write("Recognizing emergency symptoms early can save lives during natural disasters.")

    st.markdown("""
    <div class="emergency-card">
        <h4>⚠️ Red-Flag Symptoms Requiring Immediate In-Person Medical Attention:</h4>
        <ul>
            <li><strong>Severe Respiratory Distress:</strong> Continuous gasping, blue lips, or extreme chest pain.</li>
            <li><strong>Severe Dehydration:</strong> Inability to keep fluids down, dark urine or no urination for >12 hours, extreme confusion or lethargy.</li>
            <li><strong>Heat Stroke:</strong> High body temperature (>103°F/39.4°C), altered mental state, hot dry skin or profuse sweating, loss of consciousness.</li>
            <li><strong>Severe Infections / Outbreaks:</strong> High persistent fever with stiff neck, persistent vomiting, uncontrollable bloody diarrhea.</li>
            <li><strong>Water Safety / Poisoning:</strong> Sudden neurological symptoms, severe abdominal rigidity after ingesting contaminated water.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🏥 Seeking Medical Help in Pakistan")
    st.info("""
    If you encounter red-flag warning signs:
    1. Visit the nearest District Headquarter (DHQ) Hospital, Tehsil Headquarter (THQ) Hospital, or Rural Health Center (RHC).
    2. Contact official emergency rescue services operating in your province (e.g., Rescue 1122).
    3. Follow local district administration safety broadcasts.
    """)

# ==========================================
# PAGE 4: PREPAREDNESS
# ==========================================
elif navigation == "🎒 Preparedness":
    st.markdown("## 🎒 Disaster Preparedness Checklists")
    st.write("Select a disaster scenario to view standard emergency health preparedness protocols.")

    prep_disaster = st.selectbox(
        "Select Disaster Scenario",
        ["🌊 Flood Preparedness", "☀️ Heatwave Preparedness", "🦟 Dengue / Malaria Control", "💧 Safe Drinking Water"]
    )

    if prep_disaster == "🌊 Flood Preparedness":
        st.markdown("""
        - [ ] **Clean Water Reserve:** Store sealed bottled water or maintain household water purification tablets (e.g., Aquatabs).
        - [ ] **Emergency Health Kit:** ORS packets, antiseptic liquids, sterile bandages, zinc supplements, and essential chronic medications.
        - [ ] **Vector Control:** Mosquito nets, insect repellent containing DEET or Picaridin for stagnant water exposure.
        - [ ] **Document Protection:** Place medical records and family identity cards in waterproof bags.
        """)
    elif prep_disaster == "☀️ Heatwave Preparedness":
        st.markdown("""
        - [ ] **Hydration Supplies:** Adequate drinking water, Oral Rehydration Salts (ORS), and traditional cooling fluids.
        - [ ] **Indoor Cooling:** Keep living areas shaded during peak heat hours (11:00 AM - 4:00 PM).
        - [ ] **Vulnerable Care Plan:** Active monitoring of elderly, infants, and individuals with underlying chronic medical conditions.
        - [ ] **Cooling Items:** Wet towels, handheld fans, and light cotton clothing.
        """)
    elif prep_disaster == "🦟 Dengue / Malaria Control":
        st.markdown("""
        - [ ] **Eliminate Stagnant Water:** Drain standing water from buckets, tires, flowerpots, and open containers weekly.
        - [ ] **Personal Protection:** Wear long-sleeved clothing and apply repellent, especially during early morning and late afternoon peak mosquito hours.
        - [ ] **Window / Door Screens:** Ensure living areas are fitted with intact wire mesh screens.
        """)
    else:
        st.markdown("""
        - [ ] **Boiling Protocol:** Boil water vigorously for at least 1 full minute before consumption.
        - [ ] **Safe Storage:** Store treated water in clean, covered containers with narrow necks to prevent recontamination.
        - [ ] **Hygiene Maintenance:** Maintain strict handwashing with soap before preparing food and after sanitation facility use.
        """)

# ==========================================
# PAGE 5: SOURCES
# ==========================================
elif navigation == "📚 Sources":
    st.markdown("## 📚 Knowledge Base & Sources")
    st.write("SehatAI Pakistan relies strictly on authoritative documents loaded into the vector database.")

    st.markdown("""
    ### Current Active Knowledge Base
    The RAG system indexes verified guidelines from official public health and disaster authorities:
    - **World Health Organization (WHO):** Technical guidance for water safety, floods, heatwaves, and vector-borne diseases.
    - **National Health Authorities:** Outbreak prevention manuals and clinical management guidelines for Dengue and Malaria.
    - **Disaster Management Authorities:** Community emergency health and sanitation instructions.

    #### Verification & Anti-Hallucination
    - All responses are grounded in retrieved text chunks.
    - If information is not found within the loaded files, the AI explicitly alerts the user rather than fabricating details.
    """)

# ==========================================
# PAGE 6: ABOUT
# ==========================================
elif navigation == "ℹ️ About":
    st.markdown("## ℹ️ About SehatAI Pakistan")
    st.markdown("""
    **SehatAI Pakistan** is an AI-powered public health assistant designed to aid communities in Pakistan during disasters and disease outbreaks.
    
    ### Architecture Highlights
    - **Multi-Agent Design:** Division of labor across 5 distinct logical agents (Situation Analysis, RAG Search, Safety Check, Health Guidance, and Preparedness).
    - **Grounded RAG:** Uses HuggingFace Embeddings and FAISS to ground LLM generations directly in official health PDFs.
    - **Privacy-First:** No personal identifiable information (PII) or exact street addresses are requested or stored.
    - **Zero-Cost Infrastructure:** Runs completely on free-tier Streamlit Community Cloud and Groq API.
    """)

    st.divider()
    st.caption("Built for AI Hackathons | Pakistan Public Health AI Initiative")
