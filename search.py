"""
Search Module (search.py)
-------------------------
This module integrates with the Tavily Search API to perform real-time internet searches.
It handles:
1. Retrieval of the Tavily API key from environment variables.
2. Initializing the TavilyClient client wrapper.
3. Conducting searches for real-time information retrieval (e.g. current news, stock prices).
4. Formatting search results into a clean text block that can be appended to the LLM context.
5. Handling error scenarios such as missing/invalid keys, rate limits, or network failures.

Author: Antigravity AI
Date: June 2026
"""

import os
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

def get_tavily_client():
    """
    Initializes and returns the TavilyClient instance.
    Raises ValueError if the TAVILY_API_KEY environment variable is not set.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "YOUR_TAVILY_API_KEY_HERE":
        logger.error("Tavily API key is missing or set to placeholder.")
        raise ValueError("Tavily API key is not configured. Please add your GEMINI_API_KEY and TAVILY_API_KEY to the .env file.")
    return TavilyClient(api_key=api_key)

def search_tavily(query: str, max_results: int = 5) -> str:
    """
    Executes a search query using the Tavily Search API.
    Returns a formatted string containing search results (titles, snippets, and source URLs).
    Returns an error message if the search fails.
    """
    logger.info(f"Initiating Tavily search for query: '{query}'")
    try:
        client = get_tavily_client()
        # Execute search
        response = client.search(
            query=query, 
            max_results=max_results, 
            search_depth="advanced",
            include_answer=True
        )
        
        results = response.get("results", [])
        if not results:
            logger.warning(f"No search results returned for query: '{query}'")
            return "No real-time information was found for this query."
            
        # Format the search results into a clean block of context
        formatted_results = []
        formatted_results.append(f"### Real-Time Search Results for: '{query}'\n")
        
        # If Tavily provided a synthesized answer, include it as a summary
        tavily_answer = response.get("answer")
        if tavily_answer:
            formatted_results.append(f"**Tavily Summary**: {tavily_answer}\n")
            
        formatted_results.append("**Search References:**\n")
        for i, res in enumerate(results, 1):
            title = res.get("title", "Untitled Source")
            snippet = res.get("content", "No content description available.")
            url = res.get("url", "No URL provided")
            formatted_results.append(
                f"{i}. **{title}**\n"
                f"   - **Information**: {snippet}\n"
                f"   - **Source**: {url}\n"
            )
            
        context_string = "\n".join(formatted_results)
        logger.info(f"Tavily search completed. Formatted context generated ({len(results)} references).")
        return context_string
        
    except ValueError as ve:
        # Occurs when the API key is missing or not configured
        logger.error(f"Configuration error: {ve}")
        return f"[SEARCH ERROR] Configuration error: {ve}"
    except Exception as e:
        # Handles connection issues, quota issues, API rate limits, or unexpected errors
        error_msg = str(e)
        logger.error(f"Failed to execute search on Tavily: {error_msg}")
        if "401" in error_msg or "Unauthorized" in error_msg:
            return "[SEARCH ERROR] Invalid Tavily API key. Please check your credentials in the .env file."
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            return "[SEARCH ERROR] Tavily search quota or rate limit exceeded. Please try again later."
        else:
            return f"[SEARCH ERROR] An error occurred while fetching real-time data: {error_msg}"

# Run a self-test when search.py is run directly
if __name__ == "__main__":
    print("Running search module self-test (requires valid TAVILY_API_KEY)...")
    try:
        # Test search with a generic topic
        query = "latest price of Bitcoin"
        results = search_tavily(query, max_results=3)
        print(f"\nSearch output test:\n{results}")
    except Exception as ex:
        print(f"Self-test encountered an exception: {ex}")
