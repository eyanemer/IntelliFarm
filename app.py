import streamlit as st
import os
import tempfile
import speech_recognition as sr
from gtts import gTTS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import datetime
import json
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from streamlit_mic_recorder import mic_recorder
import io

# ---------------- CONFIG ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------- INITIALISATION ----------------
genai_configured = False

def configure_genai():
    global genai_configured
    if not genai_configured:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        genai_configured = True

# ---------------- FONCTIONS PDF & VECTEURS ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("vector_store")
    return vector_store

# ---------------- FONCTION CHAINE CONVERSATION ----------------
def get_conversational_chain():
    prompt_template = """
Vous êtes un expert en agriculture intelligente et IoT agricole. Vous aidez les agriculteurs et agronomes 
à optimiser leurs pratiques grâce à la technologie et aux données.

RÉGLES À SUIVRE:
1. Répondez de manière détaillée et pratique en vous basant sur le contexte fourni
2. Si l'information n'est pas dans le contexte, indiquez-le clairement
3. Proposez des conseils pratiques adaptés aux besoins agricoles
4. Expliquez les concepts techniques en termes simples
5. Mentionnez les avantages des solutions IoT pour l'agriculture
6. Soyez encourageant et positif dans vos recommandations

CONTEXTE (documents fournis):
{context}

QUESTION:
{question}

RÉPONSE DE L'EXPERT AGRICOLE:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# ---------------- FONCTION RECONNAISSANCE VOIX AVANCÉE ----------------
def recognize_speech_advanced():
    recognizer = sr.Recognizer()
    
    # Interface d'enregistrement
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🎤 Enregistrement Vocal")
        
        # Bouton d'enregistrement
        if st.button("⏺️ Démarrer l'enregistrement", use_container_width=True, type="primary"):
            with st.spinner("🔴 Enregistrement en cours... Parlez maintenant "):
                try:
                    with sr.Microphone() as source:
                        # Ajustement du bruit ambiant
                        recognizer.adjust_for_ambient_noise(source, duration=1)
                        
                        # Barre de progression pour les 15 secondes
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Enregistrement avec limite de temps
                        start_time = time.time()
                        audio = None
                        
                        for i in range(15):
                            status_text.text(f"⏱️ Temps restant: {15-i} secondes")
                            progress_bar.progress((i+1)/15)
                            try:
                                audio = recognizer.listen(source, timeout=1, phrase_time_limit=15)
                                break
                            except sr.WaitTimeoutError:
                                if time.time() - start_time >= 14:
                                    break
                                continue
                        
                        if audio:
                            status_text.text("🔍 Reconnaissance vocale en cours...")
                            # Reconnaissance vocale
                            text = recognizer.recognize_google(audio, language="fr-FR")
                            st.success("✅ Enregistrement réussi!")
                            st.session_state.recorded_question = text
                            st.session_state.show_recorded_question = True
                            return text
                        else:
                            st.warning("⏹️ Aucun son n'a été détecté. Veuillez réessayer.")
                            return None
                            
                except sr.UnknownValueError:
                    st.error("❌ Je n'ai pas compris l'audio. Veuillez parler plus clairement.")
                    return None
                except sr.RequestError as e:
                    st.error(f"❌ Erreur avec le service de reconnaissance vocale: {e}")
                    return None
                except Exception as e:
                    st.error(f"❌ Une erreur s'est produite: {e}")
                    return None
    
    return None

# ---------------- FONCTION TTS ----------------
def play_tts(answer, lang="fr"):
    try:
        with st.status("🔊 Génération de l'audio en cours...", expanded=True) as status:
            tts = gTTS(text=answer, lang=lang, slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                status.update(label="✅ Audio généré avec succès!", state="complete")
                st.audio(tmp_file.name, format="audio/mp3")
                with open(tmp_file.name, "rb") as file:
                    st.download_button(
                        label="📥 Télécharger l'audio",
                        data=file,
                        file_name="reponse_assistant.mp3",
                        mime="audio/mp3",
                        use_container_width=True
                    )
    except Exception as e:
        st.error(f"Erreur lors de la génération audio: {e}")

# ---------------- FONCTION PRINCIPALE DU CHAT ----------------
def ask_question(user_question, vector_store, chain):
    if not vector_store or not chain:
        return "⚠ Aucun document analysé. Veuillez importer et analyser vos PDF d'abord."
    docs = vector_store.similarity_search(user_question)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response["output_text"]
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({"question": user_question, "answer": answer, "time": timestamp})
    return answer

# ---------------- Fonctions pour les questions suggérées et données factices ----------------
def generate_agriculture_questions():
    return [
        "Quels capteurs IoT sont les plus utiles pour monitorer les cultures?",
        "Comment optimiser l'irrigation avec les données des capteurs?",
        "Quelles sont les meilleures pratiques pour réduire la consommation d'eau?",
        "Comment détecter précocement les maladies des plantes?",
        "Quels sont les avantages de l'agriculture de précision?",
        "Comment interpréter les données de mes capteurs sol?",
        "Quelles cultures sont les plus adaptées à mon type de sol?",
        "Comment automatiser le contrôle de l'irrigation?"
    ]

def create_mock_sensor_data():
    dates = pd.date_range(start="2023-06-01", end="2023-06-30", freq="D")
    return pd.DataFrame({
        "date": dates,
        "temperature": np.random.normal(25, 5, len(dates)),
        "humidity": np.random.normal(60, 15, len(dates)),
        "soil_moisture": np.random.normal(45, 10, len(dates)),
        "light_intensity": np.random.normal(8000, 2000, len(dates))
    })

def display_sensor_charts():
    data = create_mock_sensor_data()
    st.subheader("📊 Données des Capteurs IoT (Simulation)")
    tab1, tab2, tab3 = st.tabs(["Température & Humidité", "Humidité du Sol", "Intensité Lumineuse"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['date'], y=data['temperature'], name='Température (°C)', line=dict(color='firebrick', width=2)))
        fig.add_trace(go.Scatter(x=data['date'], y=data['humidity'], name='Humidité (%)', line=dict(color='royalblue', width=2), yaxis='y2'))
        fig.update_layout(
            title="Température et Humidité",
            xaxis_title="Date",
            yaxis=dict(title="Température (°C)"),
            yaxis2=dict(title="Humidité (%)", overlaying="y", side="right"),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['date'], y=data['soil_moisture'], name='Humidité du Sol (%)', line=dict(color='green', width=2), fill='tozeroy'))
        fig.update_layout(title="Humidité du Sol", xaxis_title="Date", yaxis_title="Humidité (%)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=data['date'], y=data['light_intensity'], name='Intensité Lumineuse', marker_color='orange'))
        fig.update_layout(title="Intensité Lumineuse", xaxis_title="Date", yaxis_title="Lux", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- THÈME AMÉLIORÉ ----------------
def apply_theme(theme):
    if theme == "Sombre":
        st.markdown(
            """
        <style>
        /* ===== GLOBAL ===== */
        .stApp { 
            background: linear-gradient(160deg, #0D1B2A 0%, #1B263B 100%);
            font-family: 'Poppins', sans-serif;
            color: #E0E1DD;
        }
        
        /* ===== BOUTONS ===== */
        .stButton>button {
            background: linear-gradient(45deg, #415A77, #778DA9);
            color: white;
            border-radius: 12px;
            border: none;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stButton>button:hover {
            background: linear-gradient(45deg, #778DA9, #415A77);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }
        
        /* ===== CARDS MÉTRIQUES ===== */
        .metric-card {
            background: linear-gradient(135deg, #1B263B 0%, #415A77 100%);
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            border: 1px solid #778DA9;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 20px rgba(0,0,0,0.3);
        }
        
        /* ===== HEADER ===== */
        header[data-testid="stHeader"] {
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            color: white;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.6);
        }

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1B263B 0%, #0D1B2A 100%);
            color: #E0E1DD;
            border-radius: 0 20px 20px 0;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            border-right: 2px solid #4CAF50;
        }
        
        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: #1B263B;
            border-radius: 12px 12px 0 0;
            padding: 12px 20px;
            font-weight: 600;
            border: 1px solid #415A77;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            color: white;
        }
        
        /* ===== INPUTS ===== */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background: #1B263B;
            border: 2px solid #415A77;
            border-radius: 12px;
            color: #E0E1DD;
            padding: 12px;
        }
        
        /* ===== RADIO BUTTONS ===== */
        .stRadio [role="radiogroup"] {
            background: #1B263B;
            padding: 15px;
            border-radius: 12px;
            border: 2px solid #415A77;
        }
        
        /* ===== EXPANDER ===== */
        .streamlit-expanderHeader {
            background: linear-gradient(90deg, #1B263B 0%, #415A77 100%);
            color: #E0E1DD;
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* ===== TITRES ===== */
        h1 { 
            color: #4CAF50; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            font-size: 2.5rem;
        }
        h2, h3 { 
            color: #8BC34A; 
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 8px;
        }
        
        /* ===== ANIMATIONS ===== */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0px); }
        }
        
        .metric-card {
            animation: float 3s ease-in-out infinite;
        }
        
        /* ===== EFFET DE BRILLANCE ===== */
        .sidebar-header {
            text-align: center;
            padding: 20px 0;
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .sidebar-header h1 {
            color: white;
            margin: 0;
            font-size: 2rem;
        }
        
        .sidebar-header p {
            color: #E0E1DD;
            margin: 5px 0 0 0;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
    else:  # Thème clair amélioré
        st.markdown(
            """
        <style>
        /* ===== GLOBAL ===== */
        .stApp { 
            background: linear-gradient(160deg, #F8F6F0 0%, #E8F5E8 100%);
            font-family: 'Poppins', sans-serif;
            color: #2E4C2E;
        }
        
        /* ===== CORRECTION COULEUR TEXTE ===== */
        .stApp, .stApp p, .stApp div, .stApp span, .stApp label {
            color: #2E4C2E !important;
        }
        
        /* ===== BOUTONS ===== */
        .stButton>button {
            background: linear-gradient(45deg, #4CAF50, #8BC34A);
            color: white;
            border-radius: 12px;
            border: none;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stButton>button:hover {
            background: linear-gradient(45deg, #8BC34A, #4CAF50);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* ===== CARDS MÉTRIQUES ===== */
        .metric-card {
            background: linear-gradient(135deg, #E8F5E8 0%, #D0E8D0 100%);
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            border: 2px solid #8BC34A;
            transition: all 0.3s ease;
            color: #2E4C2E !important;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 20px rgba(0,0,0,0.15);
        }
        
        /* ===== HEADER ===== */
        header[data-testid="stHeader"] {
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            color: white;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        }
        
        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #E8F5E8 0%, #D0E8D0 100%);
            color: #2E4C2E !important;
            border-radius: 0 20px 20px 0;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-right: 2px solid #4CAF50;
        }
        
        /* ===== CORRECTION COULEUR TEXTE SIDEBAR ===== */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #2E4C2E !important;
        }
        
        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: #E8F5E8;
            border-radius: 12px 12px 0 0;
            padding: 12px 20px;
            font-weight: 600;
            border: 2px solid #8BC34A;
            color: #2E4C2E !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            color: white !important;
        }
        
        /* ===== INPUTS ===== */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background: #FFFFFF;
            border: 2px solid #8BC34A;
            border-radius: 12px;
            color: #2E4C2E !important;
            padding: 12px;
        }
        
        /* ===== RADIO BUTTONS ===== */
        .stRadio [role="radiogroup"] {
            background: #E8F5E8;
            padding: 15px;
            border-radius: 12px;
            border: 2px solid #8BC34A;
            color: #2E4C2E !important;
        }
        
        .stRadio label {
            color: #2E4C2E !important;
        }
        
        /* ===== EXPANDER ===== */
        .streamlit-expanderHeader {
            background: linear-gradient(90deg, #E8F5E8 0%, #D0E8D0 100%);
            color: #2E4C2E !important;
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid #8BC34A;
        }
        
        /* ===== TITRES ===== */
        h1 { 
            color: #4CAF50 !important; 
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            font-size: 2.5rem;
        }
        h2, h3 { 
            color: #689F38 !important; 
            border-bottom: 2px solid #8BC34A;
            padding-bottom: 8px;
        }
        
        /* ===== ANIMATIONS ===== */
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
            40% {transform: translateY(-10px);}
            60% {transform: translateY(-5px);}
        }
        
        .metric-card {
            animation: bounce 2s ease infinite;
        }
        
        /* ===== EFFET DE BRILLANCE ===== */
        .sidebar-header {
            text-align: center;
            padding: 20px 0;
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
        }
        
        .sidebar-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2rem;
        }
        
        .sidebar-header p {
            color: #E8F5E8 !important;
            margin: 5px 0 0 0;
        }
        
        /* ===== CORRECTION COULEUR INFO ===== */
        .stInfo {
            background-color: #E8F5E8;
            border: 1px solid #8BC34A;
            color: #2E4C2E !important;
        }
        
        /* ===== CORRECTION COULEUR WARNING ===== */
        .stWarning {
            background-color: #FFF3CD;
            border: 1px solid #FFC107;
            color: #856404 !important;
        }
        
        /* ===== CORRECTION COULEUR SUCCESS ===== */
        .stSuccess {
            background-color: #D4EDDA;
            border: 1px solid #C3E6CB;
            color: #155724 !important;
        }
        
        /* ===== CORRECTION COULEUR ERROR ===== */
        .stError {
            background-color: #F8D7DA;
            border: 1px solid #F5C6CB;
            color: #721C24 !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

# ---------------- AFFICHAGE DE L'HISTORIQUE ----------------
def display_history():
    st.markdown("### 📜 Historique des Conversations")
    if not st.session_state.history:
        st.info("Aucune conversation enregistrée.")
        return

    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("🔍 Rechercher dans l'historique", placeholder="Mot-clé...")
    with col2:
        sort_order = st.selectbox("Trier par", ["Plus récent", "Plus ancien"])

    filtered_history = st.session_state.history
    if search_term:
        filtered_history = [
            item for item in st.session_state.history
            if search_term.lower() in item['question'].lower() or search_term.lower() in item['answer'].lower()
        ]
    if sort_order == "Plus récent":
        filtered_history = filtered_history[::-1]

    for i, item in enumerate(filtered_history):
        with st.expander(f"💬 Conversation du {item['time']}", expanded=False):
            st.markdown(f"**Question :** {item['question']}")
            st.markdown(f"**Réponse :** {item['answer']}")
            colx, coly, colz = st.columns([1, 1, 2])
            with colx:
                if st.button("🗑 Supprimer", key=f"delete_{i}"):
                    st.session_state.history.remove(item)
                    st.rerun()
            with coly:
                if st.button("🔊 Écouter", key=f"audio_{i}"):
                    play_tts(item['answer'])
            with colz:
                if st.button("📋 Copier", key=f"copy_{i}"):
                    st.session_state.reuse_question = item['question']
                    st.success("Question copiée!")

    if st.button("📤 Exporter l'historique complet", use_container_width=True):
        history_json = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
        st.download_button(
            label="💾 Télécharger l'historique (JSON)",
            data=history_json,
            file_name="historique_conversations.json",
            mime="application/json",
            use_container_width=True
        )

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(
        page_title="🌾 IntelliFarm – Votre assistant agricole intelligent",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Session state init
    if "history" not in st.session_state:
        st.session_state.history = []
    if "reuse_question" not in st.session_state:
        st.session_state.reuse_question = ""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "texte"
    if "theme" not in st.session_state:
        st.session_state.theme = "Clair"
    if "processed_docs_count" not in st.session_state:
        st.session_state.processed_docs_count = 0
    if "recorded_question" not in st.session_state:
        st.session_state.recorded_question = ""
    if "show_recorded_question" not in st.session_state:
        st.session_state.show_recorded_question = False

    configure_genai()

    # Appliquer le thème courant
    apply_theme(st.session_state.theme)

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
        <div class="sidebar-header">
            <h1>🌾 IntelliFarm</h1>
            <p>Votre assistant agricole intelligent</p>
        </div>
            """,
            unsafe_allow_html=True,
        )

        # 🎨 Sélecteur de thème (RADIO)
        st.markdown("#### 🎨 Thème d'affichage")
        theme_choice = st.radio(
            "Choisissez un thème :",
            ["Clair", "Sombre"],
            index=0 if st.session_state.theme == "Clair" else 1,
            horizontal=True
        )
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

        st.markdown("---")

        # 📂 Documents
        st.markdown("### 📂 Documents techniques")
        st.info("Importez vos manuels, études ou rapports techniques")
        pdf_docs = st.file_uploader(
            "Choisir des fichiers PDF",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button("🔍 Analyser les documents", use_container_width=True, type="primary"):
            if pdf_docs:
                with st.spinner("Analyse des documents en cours..."):
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)
                    st.session_state.vector_store = create_vector_store(chunks)
                    st.session_state.chain = get_conversational_chain()
                    # Mettre à jour le compteur de documents analysés
                    st.session_state.processed_docs_count = len(pdf_docs)
                st.success(f"✅ {len(pdf_docs)} document(s) analysé(s) avec succès!")
            else:
                st.warning("Veuillez importer au moins un document.")

        st.markdown("---")

    # En-tête
    st.markdown(
        """
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1>🌾 IntelliFarm</h1>
        <p style='color: #4CAF50; font-weight: 600; font-size: 1.2rem;'>Votre assistant agricole intelligent</p>
    </div>
        """,
        unsafe_allow_html=True,
    )

    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>📄</h3><h4>Documents analysés</h4><h2>{st.session_state.processed_docs_count}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>💬</h3><h4>Conversations</h4><h2>{len(st.session_state.history)}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🌡</h3><h4>Capteurs simulés</h4><h2>4</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>⏱</h3><h4>Temps moyen/réponse</h4><h2>3s</h2></div>', unsafe_allow_html=True)

    # Onglets
    tab1, tab2, tab3 = st.tabs(["💬 Assistant Conversationnel", "📊 Tableau de Bord IoT", "📜 Historique"])

    with tab1:
        st.markdown("#### 💭 Posez votre question à l'expert")

        # Mode de saisie
        input_mode = st.radio(
            "📝 Mode de saisie",
            ["Texte", "Voix"],
            index=0 if st.session_state.input_mode == "texte" else 1,
            horizontal=True
        )
        st.session_state.input_mode = input_mode.lower()

        # Interface selon le mode choisi
        user_question = ""
        ask_btn = False
        audio_btn = False

        if st.session_state.input_mode == "texte":
            user_question = st.text_area(
                "Votre question sur l'agriculture intelligente :",
                value=st.session_state.reuse_question,
                height=120,
                placeholder="Ex: Comment optimiser l'irrigation avec l'IoT ?\nQuels capteurs utiliser pour monitorer la santé des cultures ?\nComment réduire ma consommation d'eau ?"
            )
            col_a, col_b = st.columns([1, 1])
            with col_a:
                ask_btn = st.button("🌾 Envoyer ma question", use_container_width=True, type="primary")
            with col_b:
                audio_btn = st.button("🔊 Réponse audio", use_container_width=True, disabled=not user_question, type="secondary")

        else:
            # Mode voix avancé
            st.info("🎤 Utilisez le bouton ci-dessous pour enregistrer votre question vocalement (15 secondes max)")
            
            # Enregistrement vocal
            recognized_text = recognize_speech_advanced()
            
            # Afficher la question enregistrée si elle existe
            if st.session_state.show_recorded_question and st.session_state.recorded_question:
                st.markdown("### Question enregistrée:")
                st.info(f"**{st.session_state.recorded_question}**")
                
                col_a, col_b, col_c = st.columns([1, 1, 1])
                with col_a:
                    ask_btn = st.button("🌾 Envoyer cette question", use_container_width=True, type="primary")
                with col_b:
                    audio_btn = st.button("🔊 Réécouter l'audio", use_container_width=True, type="secondary")
                    st.rerun()
                
                user_question = st.session_state.recorded_question

        # On vide la question mémorisée après affichage de l'UI
        st.session_state.reuse_question = ""

        # Traitement de la question
        current_answer = None
        if ask_btn and user_question:
            with st.spinner("🔍 Recherche de la réponse..."):
                current_answer = ask_question(user_question, st.session_state.vector_store, st.session_state.chain)

            st.markdown("---")
            st.markdown("### 📝 Réponse de l'expert")
            st.markdown(f'<div class="history-item">{current_answer}</div>', unsafe_allow_html=True)

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if st.button("🔊 Écouter cette réponse", use_container_width=True, key="listen_answer_now"):
                    play_tts(current_answer)
            with col_r2:
                if st.button("📋 Copier la réponse", use_container_width=True, key="copy_answer_now"):
                    st.code(current_answer, language="markdown")
                    st.success("Réponse copiée!")

        # Bouton "Réponse audio" (lit la question si la réponse n'est pas encore générée)
        if audio_btn and user_question and not current_answer:
            if st.session_state.history:
                play_tts(st.session_state.history[-1]["answer"])
            else:
                play_tts(user_question)

    with tab2:
        st.markdown("## 📊 Tableau de Bord des Capteurs Agricoles")
        st.info("Visualisation des données simulées de capteurs IoT agricoles")
        display_sensor_charts()
        st.markdown("### 📈 Indicateurs Clés")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Température moyenne", "25°C", "2°C", delta_color="inverse")
        with c2:
            st.metric("Humidité moyenne", "60%", "-5%")
        with c3:
            st.metric("Humidité du sol", "45%", "3%")
        with c4:
            st.metric("Ensoleillement", "8000 lux", "500 lux")

    with tab3:
        display_history()

if __name__ == "__main__":
    main()