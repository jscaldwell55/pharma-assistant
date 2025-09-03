import os
import sys
import asyncio
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- CRITICAL: This makes imports from core_logic work ---
# This adds the parent 'backend' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# ---------------------------------------------------------

# Load environment variables from a .env file (for API keys, etc.)
load_dotenv() 

# --- Import your core application logic ---
from core_logic.pinecone_vector import PineconeVectorClient
from core_logic.rag import RAGRetriever
from core_logic.conversational_agent import ConversationalAgent
from core_logic.knowledge_bridge import PharmaceuticalKnowledgeBridge, EnhancedRAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")

app = Flask(__name__)
CORS(app) # Allows the browser extension to call this API

# === SINGLETON INITIALIZATION LOGIC ===
# This logic runs only ONCE when the server starts.
# It replaces the @st.cache_resource functionality from your app.py.
def build_agent_singleton():
    logger.info("Initializing ConversationalAgent singleton...")
    vec = PineconeVectorClient(
        index_name=os.getenv("PINECONE_INDEX"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
        api_key=os.getenv("PINECONE_API_KEY"),
        namespace=os.getenv("PINECONE_NAMESPACE") or None,
    )
    logger.info("Vector client initialized for index: %s", vec.index_name)
    
    base_retriever = RAGRetriever(vector_client=vec)
    knowledge_bridge = PharmaceuticalKnowledgeBridge()
    enhanced_retriever = EnhancedRAGRetriever(base_retriever, knowledge_bridge)
    
    agent = ConversationalAgent(retriever=enhanced_retriever)
    logger.info("ConversationalAgent singleton created successfully.")
    return agent

# Create the single, shared instance of your agent
agent = build_agent_singleton()
# =====================================

# === API ENDPOINT ===
@app.route('/api/rewrite', methods=['POST'])
def handle_rewrite():
    """
    This is the main API endpoint for the browser extension.
    It expects a JSON payload with 'text' (the user query) 
    and 'history' (the conversation context).
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    user_query = data['text']
    # The extension will send the history; default to empty list if not provided
    conversation_history = data.get('history', []) 

    logger.info(f"Received query: '{user_query}' with {len(conversation_history)} turns in history.")

    try:
        # Run the agent's handle method, passing in the history
        decision = asyncio.run(agent.handle(
            user_query=user_query,
            conversation_history=conversation_history
        ))
        
        # Respond with the generated text in the format the extension expects
        return jsonify({'rewritten_text': decision.response_text})

    except Exception as e:
        logger.error(f"Error handling request: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)