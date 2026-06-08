"""
Streamlit Application (app.py)
------------------------------
This is the main entry point for the AI-Powered Conversational Assistant.
It provides a professional, highly styled chat interface containing:
1. Custom CSS injections for rich visual styling (modern dark/glassmorphic theme).
2. Sidebar controls for chat session management (Create, Load, Delete, Clear history).
3. Sidebar credentials check (status of GEMINI_API_KEY and TAVILY_API_KEY).
4. Audio settings for Speech-to-Text (Voice input dictation) and Text-to-Speech (Audio playback).
5. Robust error handling displaying clean alerts for quota exhaustion (429) or API key invalidation.
6. A responsive main chat view displaying history and dynamically rendering Tavily RAG context.

Author: Antigravity AI
Date: June 2026
"""

import streamlit as st
import uuid
import os
import logging
import re
from dotenv import load_dotenv

# Import custom modules
import memory
import chatbot

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Conversational Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI Aesthetics
# Uses modern typography, dark theme gradients, rounded cards, and smooth transitions
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #FF8008 0%, #FFC837 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sidebar Styling */
    .stSidebar {
        background-color: #0f111a !important;
        border-right: 1px solid #1f2335;
    }
    
    .stSidebar [data-testid="stSidebarNav"] {
        background-color: #0f111a !important;
    }

    /* Chat Area Customization */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid #2e303f !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
    }

    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #1a1e30 !important;
        border-left: 5px solid #FF8008 !important;
    }

    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #111422 !important;
        border-left: 5px solid #00F2FE !important;
    }

    /* Button Styling */
    .stButton>button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.3s ease !important;
        border: 1px solid #3e4461 !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF8008 0%, #FFC837 100%) !important;
        color: white !important;
        border-color: transparent !important;
        transform: scale(1.02);
    }
    
    /* Critical Delete/Clear Button Styling */
    .stButton.danger-btn>button {
        background-color: #2b1111 !important;
        border-color: #5a1e1e !important;
        color: #ff6b6b !important;
    }
    
    .stButton.danger-btn>button:hover {
        background: #e63946 !important;
        color: white !important;
        border-color: transparent !important;
    }

    /* Floating RAG Badge styling */
    .rag-badge {
        padding: 4px 10px;
        background-color: #0d2b45;
        color: #00F2FE;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #144d75;
        display: inline-block;
        margin-bottom: 8px;
    }

    /* Status indicators */
    .status-card {
        padding: 10px;
        border-radius: 10px;
        font-size: 0.85rem;
        margin-bottom: 12px;
        border: 1px solid #232d38;
    }
    
    .status-ok {
        background-color: #0f2a1d;
        color: #4ade80;
        border-color: #1b4d32;
    }
    
    .status-warn {
        background-color: #2d220f;
        color: #facc15;
        border-color: #4d3c1b;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DB Initialization -----------------
memory.init_db()

# ----------------- Audio Helpers -----------------
def record_voice_input():
    """
    Attempts to read audio from the system microphone using SpeechRecognition
    and transcribes it to text via Google Speech Recognition API.
    Returns the transcription string or None if it fails.
    """
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        
        # Access microphone
        with sr.Microphone() as source:
            st.toast("🎙️ Listening... Speak now (10 seconds max)...")
            # Adjust ambient noise to improve recognition accuracy
            r.adjust_for_ambient_noise(source, duration=1.0)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            
            st.toast("⏳ Transcribing audio into text...")
            # Use Google Web Speech API for conversion
            text = r.recognize_google(audio)
            return text
            
    except ImportError:
        st.sidebar.error("Error: `SpeechRecognition` or `PyAudio` is missing. Install them to use voice input.")
        return None
    except sr.WaitTimeoutError:
        st.toast("⚠️ Listening timed out. No speech detected.", icon="🛑")
        return None
    except sr.UnknownValueError:
        st.toast("⚠️ Could not understand the audio. Speak clearly.", icon="❓")
        return None
    except sr.RequestError as e:
        st.toast(f"⚠️ Speech Recognition service error: {e}", icon="❌")
        return None
    except Exception as e:
        st.toast(f"⚠️ Microphone access failed: {e}", icon="🔌")
        return None

def generate_tts_audio(text: str):
    """
    Converts assistant text output to a local WAV file using pyttsx3.
    Cleans markdown code, links, and stars to ensure clean pronunciation.
    Returns file path or None if TTS fails.
    """
    try:
        import pyttsx3
        
        # Clean text formatting before speaking
        clean_text = re.sub(r'[*_`#~]', '', text)             # Remove md symbols
        clean_text = re.sub(r'\[.*?\]\(.*?\)', '', clean_text) # Remove link text
        clean_text = re.sub(r'```.*?```', '[Code block omitted]', clean_text, flags=re.DOTALL) # Skip code blocks
        
        filename = "temp_voice.wav"
        
        # Handle file lock if file already exists
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                # If file is occupied, create a unique one
                filename = f"temp_voice_{str(uuid.uuid4())[:8]}.wav"
                
        # Initialize and configure pyttsx3 engine
        engine = pyttsx3.init()
        engine.setProperty('rate', 160) # Set natural talking speed
        
        # Save output
        engine.save_to_file(clean_text, filename)
        engine.runAndWait()
        return filename
    except Exception as e:
        logger.error(f"Text-to-Speech conversion failed: {e}")
        return None

# ----------------- Session State Initialization -----------------
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "voice_input_buffer" not in st.session_state:
    st.session_state.voice_input_buffer = ""
if "tts_audio_path" not in st.session_state:
    st.session_state.tts_audio_path = None
if "last_tts_message" not in st.session_state:
    st.session_state.last_tts_message = ""

# Pull list of existing sessions from database
sessions_list = memory.get_sessions()

# Auto-assign active session ID if not set
if not st.session_state.active_session_id:
    if sessions_list:
        st.session_state.active_session_id = sessions_list[0]["id"]
    else:
        # Create a default conversation session
        default_id = str(uuid.uuid4())
        memory.create_session(default_id, "Welcome Session")
        st.session_state.active_session_id = default_id
        sessions_list = memory.get_sessions()

# ----------------- SIDEBAR: CONVERSATION HISTORY & CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain.png", width=64)
    st.markdown("## Assistant Dashboard")
    
    # 1. API Key Status Panel
    st.markdown("### API Integrations Status")
    gemini_key = os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    # Check Gemini
    if gemini_key and gemini_key != "YOUR_GEMINI_API_KEY_HERE":
        st.markdown('<div class="status-card status-ok">✔️ Google Gemini API: Configured</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card status-warn">⚠️ Google Gemini API: Key Missing</div>', unsafe_allow_html=True)
        
    # Check Tavily
    if tavily_key and tavily_key != "YOUR_TAVILY_API_KEY_HERE":
        st.markdown('<div class="status-card status-ok">✔️ Tavily Search API: Configured</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card status-warn">⚠️ Tavily Search API: Key Missing</div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 2. Session Controls
    st.markdown("### Session Controls")
    
    # Column buttons for session management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            memory.create_session(new_id, "New Conversation")
            st.session_state.active_session_id = new_id
            st.session_state.voice_input_buffer = ""
            st.session_state.tts_audio_path = None
            st.rerun()
            
    with col2:
        # Style deletion button with a danger warning class
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Delete Chat", use_container_width=True):
            memory.delete_session(st.session_state.active_session_id)
            st.session_state.active_session_id = None
            st.session_state.voice_input_buffer = ""
            st.session_state.tts_audio_path = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("### Saved Conversations")
    
    # Display list of conversations in a selectbox or custom radio list
    if sessions_list:
        session_titles = {sess["id"]: sess["title"] for sess in sessions_list}
        selected_sess_id = st.radio(
            label="Select conversation to load:",
            options=list(session_titles.keys()),
            format_func=lambda x: session_titles[x],
            index=list(session_titles.keys()).index(st.session_state.active_session_id) if st.session_state.active_session_id in session_titles else 0,
            label_visibility="collapsed"
        )
        if selected_sess_id != st.session_state.active_session_id:
            st.session_state.active_session_id = selected_sess_id
            st.session_state.voice_input_buffer = ""
            st.session_state.tts_audio_path = None
            st.rerun()
    else:
        st.info("No active chat sessions.")
        
    st.markdown("---")
    
    # 3. Audio & Accessibility Settings
    st.markdown("### Audio & Voice Controls")
    
    # Text-to-speech toggle
    tts_enabled = st.toggle("🔊 Enable Audio TTS Playback", value=True)
    
    # Voice Dictation Trigger
    st.markdown("**Voice Input Dictation:**")
    voice_col1, voice_col2 = st.columns([1.2, 1])
    with voice_col1:
        if st.button("🎙️ Record Voice", use_container_width=True):
            speech_text = record_voice_input()
            if speech_text:
                st.session_state.voice_input_buffer = speech_text
                st.rerun()
    with voice_col2:
        if st.button("🧹 Clear Voice", use_container_width=True):
            st.session_state.voice_input_buffer = ""
            st.rerun()
            
    # Show voice input buffer if loaded, allowing editing before sending
    if st.session_state.voice_input_buffer:
        st.info("Review voice transcript before sending:")
        voice_text_edited = st.text_area(
            label="Speech Transcript",
            value=st.session_state.voice_input_buffer,
            height=100,
            label_visibility="collapsed"
        )
        if st.button("📤 Send Voice Message", use_container_width=True):
            st.session_state.voice_input_buffer = ""
            # Triggers insertion into conversation
            st.session_state.pending_user_message = voice_text_edited
            st.rerun()
            
    # Clear all data button
    st.markdown('<div class="danger-btn" style="margin-top:20px;">', unsafe_allow_html=True)
    if st.button("🧹 Clear Database History", use_container_width=True):
        for s in memory.get_sessions():
            memory.delete_session(s["id"])
        st.session_state.active_session_id = None
        st.session_state.voice_input_buffer = ""
        st.session_state.tts_audio_path = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------- MAIN CHAT APP PAGE -----------------
# App Title Header
st.markdown("<h1>AI-Powered Conversational Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='margin-top:-15px; color:#a0aec0; font-size:1.1rem;'>"
    "Cognitive Assistant with SQLite Session Memory and Tavily Web Grounding"
    "</p>", 
    unsafe_allow_html=True
)

# Fetch conversation history for current active session
chat_messages = memory.get_messages(st.session_state.active_session_id)

# Render Chat History
for msg in chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
# Handle incoming messages
user_query = st.chat_input("How can I assist you today?")

# Handle voice submission from session state if present
if "pending_user_message" in st.session_state and st.session_state.pending_user_message:
    user_query = st.session_state.pending_user_message
    del st.session_state.pending_user_message

# If user query is provided (either typed or spoken)
if user_query:
    # 1. Display user query in UI
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. Save user message to database
    memory.save_message(st.session_state.active_session_id, "user", user_query)
    
    # 3. Update session title if it was a default "New Conversation" or "Welcome Session"
    current_session = next((s for s in sessions_list if s["id"] == st.session_state.active_session_id), None)
    if current_session and current_session["title"] in ("New Conversation", "Welcome Session"):
        # Take the first 5 words of the query as the title
        title_words = user_query.split()[:5]
        new_title = " ".join(title_words) + ("..." if len(user_query.split()) > 5 else "")
        memory.update_session_title(st.session_state.active_session_id, new_title)
        
    # 4. Generate AI response with loader
    with st.chat_message("assistant"):
        with st.spinner("Analyzing request and searching web..."):
            response_text, search_triggered, search_context = chatbot.generate_response(
                st.session_state.active_session_id, 
                user_query
            )
            
        # Display badges and search context details if triggered
        if search_triggered:
            st.markdown('<div class="rag-badge">🔍 Tavily Search Context Injected</div>', unsafe_allow_html=True)
            if search_context and not search_context.startswith("[SEARCH ERROR]"):
                with st.expander("Show Grounded Search Reference Data"):
                    st.markdown(search_context)
            elif search_context.startswith("[SEARCH ERROR]"):
                st.error(search_context)
                
        # Output final assistant response
        st.markdown(response_text)
        
        # Save assistant response to DB
        memory.save_message(st.session_state.active_session_id, "assistant", response_text)
        
        # 5. Generate TTS Audio if enabled
        if tts_enabled:
            with st.spinner("Generating speech audio..."):
                audio_file = generate_tts_audio(response_text)
                if audio_file:
                    st.session_state.tts_audio_path = audio_file
                    st.session_state.last_tts_message = response_text
                    
        # Rerun to refresh the side panel titles and play audio if created
        st.rerun()

# Play generated TTS audio from session state if ready
# Renders the audio block at the top of the main screen so it doesn't clutter chat flows
if tts_enabled and st.session_state.tts_audio_path and os.path.exists(st.session_state.tts_audio_path):
    st.markdown("#### 🔊 Assistant Voice Output")
    st.audio(st.session_state.tts_audio_path)
