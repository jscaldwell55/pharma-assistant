# backend/api/server.py
import os
import sys
import asyncio
import logging
import gc
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from dataclasses import asdict

# --- CRITICAL: This makes imports from core_logic work ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv() 

# --- PRELOAD MODELS BEFORE IMPORTS ---
# This ensures models are loaded once at startup, not during request processing
def preload_models():
    """Preload all ML models at startup to avoid timeout during requests"""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("preloader")
    
    logger.info("=== PRELOADING MODELS AT STARTUP ===")
    
    # Force garbage collection before loading
    gc.collect()
    
    # 1. Load the embedding model for RAG (largest, most important)
    logger.info("Loading embedding model for RAG...")
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        # Use the smaller model to save memory
        model = SentenceTransformer(embed_model)
        logger.info(f"✓ Loaded embedding model: {embed_model}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
    
    # 2. Skip the cross-encoder if memory is tight
    if os.getenv("SKIP_RERANKER", "false").lower() != "true":
        try:
            from sentence_transformers import CrossEncoder
            reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            reranker = CrossEncoder(reranker_model)
            logger.info(f"✓ Loaded reranker: {reranker_model}")
        except Exception as e:
            logger.warning(f"Could not load reranker (will use lexical scoring only): {e}")
            os.environ["SKIP_RERANKER"] = "true"
    
    # 3. Force another garbage collection
    gc.collect()
    
    logger.info("=== MODEL PRELOADING COMPLETE ===")

# Run preloading immediately
preload_models()

# Now import your core logic (models should already be cached)
from core_logic.pinecone_vector import PineconeVectorClient
from core_logic.rag import RAGRetriever
from core_logic.conversational_agent import ConversationalAgent
from core_logic.knowledge_bridge import PharmaceuticalKnowledgeBridge, EnhancedRAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")

app = Flask(__name__, static_folder='../../static')

# Configure CORS
CORS(app, origins=[
    "chrome-extension://*",
    "https://pharma-assistant-api.onrender.com",
    "http://localhost:*",
    "http://127.0.0.1:*",
])

# === SINGLETON INITIALIZATION WITH ERROR HANDLING ===
agent = None

def build_agent_singleton():
    """Build the conversational agent with all components"""
    global agent
    
    if agent is not None:
        return agent
        
    logger.info("Initializing ConversationalAgent singleton...")
    
    try:
        # Initialize vector client
        vec = PineconeVectorClient(
            index_name=os.getenv("PINECONE_INDEX", "pharma-assistant"),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
            api_key=os.getenv("PINECONE_API_KEY"),
            namespace=os.getenv("PINECONE_NAMESPACE") or None,
        )
        logger.info("Vector client initialized for index: %s", vec.index_name)
        
        # Build retrieval pipeline
        base_retriever = RAGRetriever(vector_client=vec)
        knowledge_bridge = PharmaceuticalKnowledgeBridge()
        enhanced_retriever = EnhancedRAGRetriever(base_retriever, knowledge_bridge)
        
        # Create agent
        agent = ConversationalAgent(retriever=enhanced_retriever)
        logger.info("ConversationalAgent singleton created successfully.")
        
        # Force garbage collection after initialization
        gc.collect()
        
        return agent
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        # Don't raise - allow server to start even if initialization fails
        return None

# Initialize agent after app creation
# Using a flag to ensure it only happens once
_agent_initialized = False

def ensure_agent_initialized():
    """Ensure agent is initialized (called on first request)"""
    global agent, _agent_initialized
    if not _agent_initialized:
        if agent is None:
            agent = build_agent_singleton()
        _agent_initialized = True

# === HEALTH CHECK (doesn't require agent) ===
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    global agent
    status = 'ready' if agent is not None else 'initializing'
    return jsonify({'status': 'healthy', 'service': 'pharma-assistant', 'agent_status': status}), 200

# === STATIC FILE ROUTES ===
@app.route('/')
def serve_index():
    """Serve the main chat interface"""
    try:
        static_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static')
        index_path = os.path.join(static_dir, 'index.html')
        
        if not os.path.exists(index_path):
            logger.error(f"index.html not found at: {index_path}")
            return jsonify({'error': 'Chat interface not found'}), 404
            
        return send_file(index_path)
    except Exception as e:
        logger.error(f"Error serving index: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    try:
        static_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static')
        return send_from_directory(static_dir, filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}", exc_info=True)
        return jsonify({'error': 'File not found'}), 404

# === API ENDPOINTS ===
@app.route('/api/rewrite', methods=['POST'])
def handle_rewrite():
    """Original endpoint for browser extension compatibility"""
    global agent
    
    # Ensure agent is initialized
    ensure_agent_initialized()
        
    if agent is None:
        return jsonify({'error': 'Service is initializing, please try again in a moment'}), 503
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    user_query = data['text']
    conversation_history = data.get('history', [])

    logger.info(f"Received query: '{user_query[:100]}...' with {len(conversation_history)} turns")

    try:
        # Run with timeout to prevent hanging
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_with_timeout():
            return await asyncio.wait_for(
                agent.handle(
                    user_query=user_query,
                    conversation_history=conversation_history
                ),
                timeout=30.0  # 30 second timeout
            )
        
        decision = loop.run_until_complete(run_with_timeout())
        loop.close()
        
        # Force garbage collection after processing
        gc.collect()
        
        return jsonify({'rewritten_text': decision.response_text})

    except asyncio.TimeoutError:
        logger.error("Request timed out after 30 seconds")
        return jsonify({'error': 'Request timed out, please try again'}), 504
    except Exception as e:
        logger.error(f"Error handling request: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred'}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Enhanced endpoint for web UI"""
    global agent
    
    # Ensure agent is initialized
    ensure_agent_initialized()
        
    if agent is None:
        return jsonify({'error': 'Service is initializing, please try again in a moment'}), 503
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    user_query = data['text']
    conversation_history = data.get('history', [])

    logger.info(f"Received chat query: '{user_query[:100]}...'")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_with_timeout():
            return await asyncio.wait_for(
                agent.handle(
                    user_query=user_query,
                    conversation_history=conversation_history
                ),
                timeout=30.0
            )
        
        decision = loop.run_until_complete(run_with_timeout())
        loop.close()
        
        # Force garbage collection
        gc.collect()
        
        return jsonify(asdict(decision))

    except asyncio.TimeoutError:
        logger.error("Request timed out after 30 seconds")
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        logger.error(f"Error handling chat: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the agent's cache"""
    global agent
    
    if agent is None:
        return jsonify({'error': 'Agent not initialized'}), 503
    
    try:
        agent.clear_answer_cache()
        if hasattr(agent.retriever, 'clear_cache'):
            agent.retriever.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Cache cleared successfully")
        return jsonify({'success': True, 'message': 'Cache cleared'})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        return jsonify({'error': 'Failed to clear cache'}), 500

# === ERROR HANDLERS ===
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    # Check for required environment variables
    required_vars = ['PINECONE_API_KEY', 'ANTHROPIC_API_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
    
    # Run with Gunicorn in production, Flask dev server locally
    app.run(host='0.0.0.0', port=port, debug=False)