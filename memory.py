"""
Memory Module (memory.py)
-------------------------
This module provides persistent storage for the conversational assistant using SQLite.
It manages chat sessions and individual chat messages, allowing the application to:
1. Initialize the SQLite database and create necessary tables.
2. Create, retrieve, and delete chat sessions.
3. Save user messages and assistant responses.
4. Load conversation history.
5. Retrieve the last 10 messages of the active session to serve as context for the Gemini API.

Author: Antigravity AI
Date: June 2026
"""

import sqlite3
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "database.db"

def get_db_connection(db_path=DB_PATH):
    """
    Establishes and returns a connection to the SQLite database.
    Enables foreign key support.
    """
    try:
        conn = sqlite3.connect(db_path)
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON;")
        # Return rows as dictionary-like objects for easier access
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

def init_db(db_path=DB_PATH):
    """
    Initializes the SQLite database. Creates 'sessions' and 'messages' tables 
    if they do not exist.
    """
    logger.info("Initializing database...")
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Create Sessions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create Messages Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sender TEXT NOT NULL CHECK(sender IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        if conn:
            conn.close()

def create_session(session_id: str, title: str = "New Conversation", db_path=DB_PATH):
    """
    Creates a new chat session in the database.
    """
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO sessions (id, title) VALUES (?, ?)",
            (session_id, title)
        )
        conn.commit()
        logger.info(f"Session '{session_id}' created/verified.")
    except sqlite3.Error as e:
        logger.error(f"Error creating session: {e}")
    finally:
        if conn:
            conn.close()

def get_sessions(db_path=DB_PATH):
    """
    Retrieves all chat sessions sorted by creation time descending.
    """
    conn = None
    sessions = []
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, created_at FROM sessions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        for row in rows:
            sessions.append({
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"]
            })
    except sqlite3.Error as e:
        logger.error(f"Error fetching sessions: {e}")
    finally:
        if conn:
            conn.close()
    return sessions

def update_session_title(session_id: str, title: str, db_path=DB_PATH):
    """
    Updates the title of a specific chat session (e.g., set title based on the first query).
    """
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title, session_id)
        )
        conn.commit()
        logger.info(f"Updated title for session '{session_id}' to: '{title}'")
    except sqlite3.Error as e:
        logger.error(f"Error updating session title: {e}")
    finally:
        if conn:
            conn.close()

def delete_session(session_id: str, db_path=DB_PATH):
    """
    Deletes a session and all its messages (due to ON DELETE CASCADE).
    """
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        logger.info(f"Session '{session_id}' deleted successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error deleting session '{session_id}': {e}")
    finally:
        if conn:
            conn.close()

def save_message(session_id: str, sender: str, content: str, db_path=DB_PATH):
    """
    Saves a message (either user or assistant) into the database under the given session.
    """
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, sender, content) VALUES (?, ?, ?)",
            (session_id, sender, content)
        )
        conn.commit()
        logger.debug(f"Saved message from {sender} in session {session_id}")
    except sqlite3.Error as e:
        logger.error(f"Error saving message: {e}")
    finally:
        if conn:
            conn.close()

def get_messages(session_id: str, db_path=DB_PATH):
    """
    Retrieves all messages for a specific session ordered by timestamp ascending.
    """
    conn = None
    messages = []
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sender, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        for row in rows:
            messages.append({
                "role": row["sender"], # 'user' or 'assistant'
                "content": row["content"],
                "timestamp": row["timestamp"]
            })
    except sqlite3.Error as e:
        logger.error(f"Error fetching messages for session {session_id}: {e}")
    finally:
        if conn:
            conn.close()
    return messages

def get_context_messages(session_id: str, limit: int = 10, db_path=DB_PATH):
    """
    Retrieves the last N (default 10) messages of a session to act as conversational context.
    Returns them in chronological order.
    """
    conn = None
    messages = []
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        # Fetch the last N messages, ordered by timestamp desc to get the most recent ones first
        cursor.execute(
            "SELECT sender, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        # Reverse the list so it is in chronological order for the model
        for row in reversed(rows):
            messages.append({
                "role": row["sender"],
                "content": row["content"]
            })
    except sqlite3.Error as e:
        logger.error(f"Error fetching context for session {session_id}: {e}")
    finally:
        if conn:
            conn.close()
    return messages

# Run a self-test when memory.py is run directly
if __name__ == "__main__":
    test_db = "test_database.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    print("Running memory module self-test...")
    init_db(test_db)
    
    sess_id = "test-session-123"
    create_session(sess_id, "Test Conversation Title", test_db)
    
    save_message(sess_id, "user", "Hello assistant!", test_db)
    save_message(sess_id, "assistant", "Hello! How can I help you today?", test_db)
    save_message(sess_id, "user", "Can you explain SQLite?", test_db)
    save_message(sess_id, "assistant", "SQLite is a C-language library that implements a small, fast, self-contained SQL database engine.", test_db)
    
    print("\nAll Sessions:")
    for s in get_sessions(test_db):
        print(f" - {s['title']} (ID: {s['id']})")
        
    print("\nLast 2 Messages Context:")
    for m in get_context_messages(sess_id, limit=2, db_path=test_db):
        print(f" - {m['role']}: {m['content']}")
        
    print("\nFull Chat History:")
    for m in get_messages(sess_id, test_db):
        print(f" - {m['role']}: {m['content']} ({m['timestamp']})")
        
    # Test title update
    update_session_title(sess_id, "Explaining SQLite", test_db)
    print("\nUpdated Sessions:")
    for s in get_sessions(test_db):
        print(f" - {s['title']} (ID: {s['id']})")
        
    # Clean up test database
    if os.path.exists(test_db):
        os.remove(test_db)
    print("\nSelf-test finished successfully. Test database cleaned up.")
