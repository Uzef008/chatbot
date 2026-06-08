# AI-Powered Conversational Assistant with Memory and Real-Time Search

A complete, modular, and professional Python application featuring an AI assistant equipped with persistent conversation memory (SQLite) and real-time search integration (Tavily Search API) powered by the Google Gemini API. It features a custom, high-end Streamlit web interface and includes optional local Voice Input (Speech-to-Text) and Voice Output (Text-to-Speech).

This project has been developed following software engineering standards, making it ideal for final-year engineering internships and project submissions.

---

## Project Overview

Modern LLMs are highly capable but suffer from two main limitations:
1. **Knowledge Cutoff**: They cannot access real-time information (e.g. current weather, today's sports scores, stock prices).
2. **Context Window Limitations**: In long chats, models lose context. Maintaining a local persistent conversation history helps ground the conversation.

This project solves these limitations by implementing a **Retrieval-Augmented Generation (RAG)** pipeline:
- **Conversation Persistence**: An SQLite database stores all sessions and messages. It extracts the last 10 messages to construct conversational context.
- **Auto Query Classification**: A rule-based classifier automatically detects queries that require real-time information.
- **Search Grounding**: For real-time queries, it uses the Tavily Search API to retrieve internet context, which is injected into the Gemini prompt along with the chat history.

---

## Features

- 🤖 **Gemini AI Integration**: Powered by `gemini-1.5-flash` for high-speed, cost-effective responses.
- 💾 **Persistent SQLite Memory**: Stores user and assistant conversations across runs. Allows creating, renaming, and deleting conversations.
- 🔍 **Real-Time Web Search**: Injects live internet snippets from Tavily when real-time information (sports, weather, stock, crypto, news) is queried.
- 🎙️ **Voice Dictation**: Capture voice input using Python's `SpeechRecognition` library.
- 🔊 **Voice Synthesis Output**: Generate text-to-speech (TTS) audio files using `pyttsx3` and plays them directly in the Streamlit web player.
- 🛡️ **Robust Error Handlers**: Intercepts missing environment configurations, network timeouts, invalid API keys, and Gemini `429 RESOURCE_EXHAUSTED` rate limits. Displays beautiful, user-friendly error banners rather than crashing the system.
- 🎨 **Premium Glassmorphic UI**: Uses custom CSS variables, gradients, and rounded margins to construct a high-end interface.

---

## Project Structure

```text
project/
│
├── app.py              # Main Streamlit web application & custom CSS
├── chatbot.py          # Chat coordinator, query classifier, & Gemini API integration
├── memory.py           # SQLite database wrapper for persistent session & chat history
├── search.py           # Tavily search API integration wrapper
├── requirements.txt    # Project dependencies list
├── .env                # Secret environment variables (ignored by git)
├── .gitignore          # File exclusions list (excludes credentials and db)
└── README.md           # Documentation (this file)
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9 or higher (Python 3.10 is recommended).
- Internet connection (to access Google Gemini and Tavily APIs).
- System audio drivers and microphone (for Voice input and Voice output).

### 2. Clone the Repository
Extract the files into a directory (e.g., `D:/projectd/python 1`).

### 3. Create a Virtual Environment (Optional but Recommended)
Open a terminal in the project directory and run:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
Install Python libraries via pip:
```bash
python -m pip install -r requirements.txt
```

#### 🛠️ Voice Input & Text-to-Speech System Requirements
The voice libraries (`SpeechRecognition`, `PyAudio`, and `pyttsx3`) rely on system-level libraries:
- **SpeechRecognition & PyAudio**:
  - **Windows**: PyAudio is pre-compiled. If standard pip fails, run:
    ```bash
    pip install pipwin
    pipwin install pyaudio
    ```
  - **macOS**: Install PortAudio via Homebrew first:
    ```bash
    brew install portaudio
    pip install pyaudio
    ```
  - **Linux (Ubuntu/Debian)**: Install development libraries:
    ```bash
    sudo apt-get install python3-pyaudio portaudio19-dev
    pip install pyaudio
    ```
- **pyttsx3**:
  - Works out of the box on Windows (SAPI5) and macOS (NSSpeechSynthesizer).
  - On Linux, install `espeak` synthesizer:
    ```bash
    sudo apt-get install espeak
    ```

If any speech library fails to install or initialize, the application **will still run successfully** in text-only mode and display standard widgets, bypassing speech.

### 5. Setup Environment Variables
1. Obtain a Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).
2. Obtain a Tavily Search API Key from [Tavily](https://tavily.com/).
3. Open the `.env` file in the project directory and insert your keys:
```env
GEMINI_API_KEY=AIzaSy...your_gemini_key...
TAVILY_API_KEY=tvly-...your_tavily_key...
```

---

## Usage

### Run the Application Locally
Run the Streamlit application from your terminal:
```bash
streamlit run app.py
```
This command spins up a local server and automatically opens a browser window at `http://localhost:8501`.

### Operating the Interface
1. **Chatting**: Type your message in the chat input at the bottom and press Enter.
2. **Conversation Sidebar**:
   - Click **➕ New Chat** to create a fresh conversation.
   - Click **🗑️ Delete Chat** to remove the active conversation and its history.
   - Select conversation items in the sidebar radio list to load past conversations.
   - Click **🧹 Clear Database History** to wipe out the database clean.
3. **Voice Input**:
   - Click **🎙️ Record Voice** in the sidebar. Speak clearly into your microphone.
   - Once transcribed, the text will load in the sidebar text box. You can edit the text, then click **📤 Send Voice Message**.
4. **Text-to-Speech**:
   - Toggle **🔊 Enable Audio TTS Playback** in the sidebar.
   - The assistant's text answers will automatically generate a voice output player under the header after each response.

---

## Testing Guide

### 1. Test Persistent Memory
1. Start a conversation and send: `"Hi, my name is John. I am a computer science student."`
2. Close the browser tab and stop the Streamlit server in the terminal.
3. Start the server again (`streamlit run app.py`).
4. Select the prior conversation session from the sidebar.
5. Send: `"What is my name and what do I study?"`
6. **Expected Outcome**: The assistant retrieves your name and field of study using the database context.

### 2. Test Real-Time Search Grounding
1. In a conversation, query a real-time topic, such as:
   - `"What is the current price of Bitcoin today?"`
   - `"What is the weather like in Tokyo right now?"`
   - `"Give me the latest news headlines about SpaceX."`
2. **Expected Outcome**:
   - The UI displays a blue badge: `🔍 Tavily Search Context Injected`.
   - The chatbot uses Tavily search, retrieves current values, and sends them to Gemini.
   - Gemini answers with up-to-date data and cites the source URLs in the response.
   - You can click the collapsible **Show Grounded Search Reference Data** box to inspect raw search details.

### 3. Test Error Handlers
1. Open the `.env` file and break the API key (e.g. set `GEMINI_API_KEY=invalid_key_test`).
2. Reload the chat and enter a prompt.
3. **Expected Outcome**: The application does not crash. It displays a red warning panel informing you that the API key configuration is invalid and instructions on how to fix it.

---

## Deployment to Streamlit Cloud

To host this assistant on Streamlit Cloud:

1. **Commit code to GitHub**:
   Ensure `requirements.txt`, `app.py`, `chatbot.py`, `memory.py`, and `search.py` are committed. Ensure `.env` and `database.db` are **not** committed (they should be listed in `.gitignore`).

2. **Deploy on Streamlit**:
   - Connect your GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/).
   - Click **New App**, select your repository, branch, and set entry point to `app.py`.

3. **Configure Secrets**:
   Instead of uploading a `.env` file, go to your app settings on Streamlit Cloud -> **Secrets** and paste the keys:
   ```toml
   GEMINI_API_KEY = "your_google_gemini_key_here"
   TAVILY_API_KEY = "your_tavily_api_key_here"
   ```
   Streamlit Cloud will inject these into system environment variables, where our modules will automatically load them.

*Note: In the cloud environment, local audio recording via browser microphone using PyAudio/SpeechRecognition requires custom browser setups. The application will run in standard text-only mode cleanly on the cloud, displaying standard notifications if microphone features are triggered.*

---

## Future Enhancements

1. **Vector DB (ChromaDB) RAG**: Transition from retrieving only the last 10 messages to performing semantic search over the entire historical archive.
2. **Direct Browser Audio Recording**: Implement a custom Streamlit JS component (like `streamlit-mic-recorder`) to record audio inside the user's browser client and stream it to the server.
3. **Advanced LLM Classification**: Replace keyword detection with an LLM function-calling schema or zero-shot classifiers to detect real-time searches more accurately.
4. **Local LLM Offline Fallback**: Implement Ollama integration to run a local model (e.g. Llama3) if the internet connection is lost.
#   c h a t b o t  
 #   c h a t b o t  
 