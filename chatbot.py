"""
Chatbot Module (chatbot.py)
---------------------------
This module orchestrates the AI logic of the application:
1. Loads the Google Gemini API key from the environment.
2. Detects whether the user's query requires real-time information retrieval (e.g. weather, news, stocks).
3. Integrates with the search module to retrieve real-time context if needed.
4. Retrieves conversational context (last 10 messages) from memory.py.
5. Interfaces with Google Gemini API to generate responses.
6. Implements robust error handling for API keys, network issues, and 429 quota exhaustion.

Author: Antigravity AI
Date: June 2026
"""

import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Import custom modules
import memory
import search

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import Google API exceptions if available for fine-grained error handling
try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None

# Default Gemini model (can be updated dynamically based on available API key models)
MODEL_NAME = "gemini-1.5-flash"

def configure_gemini():
    """
    Configures the google-generativeai package with the API key from environment.
    Raises ValueError if key is missing or set to placeholder.
    Dynamically falls back to best available model in the API key's tier.
    """
    global MODEL_NAME
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("Gemini API key is missing or placeholder in environment.")
        raise ValueError("Google Gemini API key is not configured. Please add your GEMINI_API_KEY and TAVILY_API_KEY to the .env file.")
    genai.configure(api_key=api_key)
    
    # Dynamically query available models to select best option supported by this key
    try:
        available_models = [m.name for m in genai.list_models()]
        model_priority = [
            "models/gemini-3.5-flash",
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash",
            "models/gemini-flash-latest",
            "models/gemini-pro-latest"
        ]
        for candidate in model_priority:
            if candidate in available_models:
                MODEL_NAME = candidate.replace("models/", "")
                logger.info(f"Dynamically selected active Gemini model: {MODEL_NAME}")
                break
    except Exception as e:
        logger.warning(f"Could not dynamically list models, defaulting to standard model {MODEL_NAME}: {e}")

def detect_real_time_query(query: str) -> bool:
    """
    Analyzes the user's query to determine if it requires real-time search lookup.
    Checks for keywords related to: weather, news, sports scores, stock prices, bitcoin, etc.
    """
    query_lower = query.lower().strip()
    
    # Keyword list categorized by requirements
    real_time_keywords = [
        # Weather
        "weather", "temperature", "forecast", "temp", "rain", "snow", "humidity", "wind speed",
        # Bitcoin/Crypto
        "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "dogecoin", "cryptocurrency",
        # Latest News
        "news", "headline", "breaking", "latest update", "happened today", "current events",
        # Sports Scores
        "score", "match", "standing", "points table", "won the game", "sports", "cricket", "football", 
        "soccer", "nfl", "nba", "mlb", "fifa", "premier league", "championship",
        # Stock Prices & Financials
        "stock", "ticker", "pe ratio", "market cap", "nasdaq", "nyse", "share price", "share value", "stock market",
        # Time-sensitive / Current modifiers
        "current", "latest", "recent", "today", "now", "yesterday", "who is the current", "what is the current"
    ]
    
    # Check if any keyword matches
    is_realtime = any(keyword in query_lower for keyword in real_time_keywords)
    if is_realtime:
        logger.info(f"Query '{query}' classified as REAL-TIME query.")
    else:
        logger.info(f"Query '{query}' classified as STANDARD query.")
    return is_realtime

def build_gemini_history(context_messages: list) -> list:
    """
    Converts list of messages from database to Gemini chat history format.
    Database format has roles: 'user', 'assistant'.
    Gemini format requires roles: 'user', 'model'.
    """
    gemini_history = []
    for msg in context_messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({
            "role": role,
            "parts": [msg["content"]]
        })
    return gemini_history

def generate_response(session_id: str, user_query: str) -> tuple:
    """
    Orchestrates response generation:
    1. Checks if search is needed and pulls search context.
    2. Pulls chat history context (last 10 messages) from database.
    3. Triggers Gemini API with custom prompt and history.
    4. Handles API and configuration exceptions.
    
    Returns: (response_text, search_triggered_flag, search_results_text)
    """
    search_triggered = False
    search_results = ""
    
    # 1. Check config and initialize Gemini
    try:
        configure_gemini()
    except ValueError as ve:
        return str(ve), False, ""
        
    # 2. Detect if real-time query and perform search if true
    if detect_real_time_query(user_query):
        search_triggered = True
        search_results = search.search_tavily(user_query)
        
    # 3. Fetch chat history (last 10 messages) to maintain conversational context
    # This history will exclude the current user_query which is sent live
    history_context = memory.get_context_messages(session_id, limit=10)
    gemini_history = build_gemini_history(history_context)
    
    # 4. Construct system instruction & direct input
    system_instruction = (
        "You are a helpful, professional, and knowledgeable AI conversational assistant. "
        "You are equipped with a persistent SQLite memory of the current chat session and a real-time search engine. "
        "Always maintain a natural, engaging, and friendly tone. "
        "When real-time search context is provided, prioritize it to answer current topics, news, or figures, "
        "and make sure to reference/cite the sources (such as the website name and link) organically. "
        "If the search results are incomplete or unavailable, explain this and use your training data, "
        "making clear which information is real-time and which is from your knowledge cutoff."
    )
    
    # Prepare prompt with search context if applicable
    if search_triggered and not search_results.startswith("[SEARCH ERROR]"):
        prompt = (
            f"Use the following real-time search results to answer the user's question:\n\n"
            f"{search_results}\n\n"
            f"User Question: {user_query}\n\n"
            f"Generate a clear, accurate response based on these search results and the prior chat context. "
            f"Cite the source URLs provided above."
        )
    else:
        # If search failed or was not triggered
        prompt = user_query
        
    # 5. Call Gemini API with generation settings
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction
        )
        
        # Start chat with loaded history
        chat = model.start_chat(history=gemini_history)
        
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=1024
        )
        
        logger.info("Sending message to Gemini API...")
        response = chat.send_message(prompt, generation_config=generation_config)
        response_text = response.text
        logger.info("Response received successfully from Gemini API.")
        
        return response_text, search_triggered, search_results
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Gemini API invocation failed: {error_msg}")
        
        # Check for specific quota limit exceptions
        if google_exceptions and isinstance(e, google_exceptions.ResourceExhausted):
            user_friendly_error = (
                "⚠️ **API Quota Exceeded (429 RESOURCE_EXHAUSTED)**\n\n"
                "The Gemini API rate limit or quota has been temporarily reached. "
                "Please wait a moment and try sending your message again. "
                "If you are using a free tier API key, consider upgrading or checking your Google AI Studio dashboard."
            )
        elif google_exceptions and isinstance(e, google_exceptions.InvalidArgument):
            user_friendly_error = (
                "⚠️ **Invalid API Key Configuration**\n\n"
                "The Google Gemini API key provided in the `.env` file appears to be invalid or deactivated. "
                "Please double-check your credentials."
            )
        elif "429" in error_msg or "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
            user_friendly_error = (
                "⚠️ **API Quota Exceeded (429 RESOURCE_EXHAUSTED)**\n\n"
                "The Gemini API rate limit or quota has been temporarily reached. "
                "Please wait a moment and try sending your message again."
            )
        elif "400" in error_msg or "invalid" in error_msg.lower() or "key" in error_msg.lower():
            user_friendly_error = (
                "⚠️ **API Key Error**\n\n"
                "An error occurred indicating that the API key is invalid or unauthorized. "
                "Please update your `.env` file with a valid Gemini API key from Google AI Studio."
            )
        elif "connection" in error_msg.lower() or "dns" in error_msg.lower() or "network" in error_msg.lower():
            user_friendly_error = (
                "⚠️ **Network Connection Failure**\n\n"
                "Unable to connect to Google API servers. Please check your internet connection and try again."
            )
        else:
            user_friendly_error = (
                f"⚠️ **An Error Occurred**\n\n"
                f"The system encountered an error while communicating with Gemini API:\n"
                f"`{error_msg}`"
            )
            
        return user_friendly_error, search_triggered, search_results

# Run a self-test when chatbot.py is run directly
if __name__ == "__main__":
    print("Running chatbot module self-test...")
    # Setup test memory
    memory.init_db("test_chatbot.db")
    test_session = "test-session-456"
    memory.create_session(test_session, "Self Test Session", "test_chatbot.db")
    
    # 1. Test Query Detection
    print("\nTesting query detection:")
    queries = [
        "What is the weather like in New York today?",
        "Who is the current President of France?",
        "How does a database work?",
        "Bitcoin price chart ticker",
        "Write a poem about stars"
    ]
    for q in queries:
        print(f" - Query: '{q}' | Needs search? {detect_real_time_query(q)}")
        
    # Clean up test database
    if os.path.exists("test_chatbot.db"):
        os.remove("test_chatbot.db")
    print("\nSelf-test finished successfully.")
