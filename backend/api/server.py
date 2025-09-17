# backend/api/server_lightweight.py
"""
Lightweight Flask server for Render that delegates ML operations to Modal.
This replaces the heavy server.py with a memory-efficient version.
"""

import os
import sys
import asyncio
import logging
import gc
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from dataclasses import asdict

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Import only the lightweight components
from core_logic.pinecone_modal_client import PineconeModalClient
from core_logic.rag import RAGRetriever  # Modified to use Modal client
from core_logic.conversational_agent import ConversationalAgent
from core_logic.knowledge_bridge import PharmaceuticalKnowledgeBridge, EnhancedRAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server_light")

app = Flask(__name__, static_folder='../../static')

# Configure CORS
CORS(app, origins=[
    "chrome-extension://*",
    "https://pharma-assistant-api.onrender.com",
    "http://localhost:*",
    "http://127.0.0.1:*",
])

# Global agent (lazy initialization)
agent = None

def build_agent_singleton():
    """Build the conversational agent with Modal-based components"""
    global agent
    
    if agent is not None:
        return agent
    
    logger.info("Initializing lightweight agent with Modal embedder...")
    
    try:
        # Check for Modal endpoint
        modal_endpoint = os.getenv("MODAL_EMBEDDER_ENDPOINT")
        if not modal_endpoint:
            logger.error("MODAL_EMBEDDER_ENDPOINT not set!")
            logger.info("Deploy Modal service first: modal deploy modal_embedder.py")
            logger.info("Then set MODAL_EMBEDDER_ENDPOINT to the URL Modal provides")
            return None
        
        # Initialize Pinecone with Modal embedder
        vec_client = PineconeModalClient(
            index_name=os.getenv("PINECONE_INDEX", "pharma-assistant"),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
            api_key=os.getenv("PINECONE_API_KEY"),
            namespace=os.getenv("PINECONE_NAMESPACE") or None,
            modal_endpoint=modal_endpoint,
            modal_api_key=os.getenv("MODAL_API_KEY")  # Optional
        )
        logger.info("Pinecone+Modal client initialized")
        
        # Build retrieval pipeline
        base_retriever = RAGRetriever(vector_client=vec_client)
        knowledge_bridge = PharmaceuticalKnowledgeBridge()
        enhanced_retriever = EnhancedRAGRetriever(base_retriever, knowledge_bridge)
        
        # Create agent
        agent = ConversationalAgent(retriever=enhanced_retriever)
        logger.info("✅ Lightweight agent initialized successfully")
        
        # No heavy models to load - everything is delegated to Modal!
        gc.collect()
        
        return agent
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        return None

def ensure_agent_initialized():
    """Ensure agent is initialized"""
    global agent
    if agent is None:
        agent = build_agent_singleton()
    return agent is not None

# === HEALTH CHECK ===
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    modal_endpoint = os.getenv("MODAL_EMBEDDER_ENDPOINT", "not_set")
    agent_ready = agent is not None
    
    return jsonify({
        'status': 'healthy',
        'service': 'pharma-assistant-lightweight',
        'agent_ready': agent_ready,
        'modal_configured': modal_endpoint != "not_set",
        'modal_endpoint': modal_endpoint if modal_endpoint != "not_set" else None
    }), 200

# === MAIN ENDPOINTS ===
@app.route('/api/rewrite', methods=['POST'])
def handle_rewrite():
    """Browser extension compatibility endpoint"""
    if not ensure_agent_initialized():
        return jsonify({'error': 'Service not ready. Check Modal configuration.'}), 503
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    user_query = data['text']
    conversation_history = data.get('history', [])
    
    logger.info(f"Query: '{user_query[:100]}...'")
    
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
        
        gc.collect()
        
        return jsonify({'rewritten_text': decision.response_text})
        
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Web UI endpoint"""
    if not ensure_agent_initialized():
        return jsonify({'error': 'Service not ready. Check Modal configuration.'}), 503
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    user_query = data['text']
    conversation_history = data.get('history', [])
    
    logger.info(f"Chat query: '{user_query[:100]}...'")
    
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
        
        gc.collect()
        
        return jsonify(asdict(decision))
        
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/test-modal', methods=['GET'])
def test_modal():
    """Test Modal embedding service connectivity"""
    import requests
    
    modal_endpoint = os.getenv("MODAL_EMBEDDER_ENDPOINT", "")
    
    if not modal_endpoint:
        return jsonify({
            'status': 'error',
            'message': 'MODAL_EMBEDDER_ENDPOINT not configured'
        }), 400
    
    try:
        # Test Modal service
        response = requests.get(
            f"{modal_endpoint}/get_info",
            timeout=10
        )
        response.raise_for_status()
        info = response.json()
        
        # Test embedding
        test_response = requests.post(
            f"{modal_endpoint}/embed_single",
            json={"text": "test", "normalize": True},
            timeout=10
        )
        test_response.raise_for_status()
        embedding = test_response.json()
        
        return jsonify({
            'status': 'success',
            'modal_info': info,
            'test_embedding_dims': len(embedding) if isinstance(embedding, list) else 0
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'modal_endpoint': modal_endpoint
        }), 500

# === STATIC FILES ===
@app.route('/')
def serve_index():
    """Serve main interface"""
    try:
        static_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static')
        index_path = os.path.join(static_dir, 'index.html')
        
        if not os.path.exists(index_path):
            return jsonify({'error': 'Interface not found'}), 404
        
        return send_file(index_path)
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return jsonify({'error': 'Internal error'}), 500

# === ERROR HANDLERS ===
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    # Check configuration
    if not os.getenv("MODAL_EMBEDDER_ENDPOINT"):
        logger.warning("=" * 60)
        logger.warning("MODAL_EMBEDDER_ENDPOINT not set!")
        logger.warning("Deploy Modal first: modal deploy modal_embedder.py")
        logger.warning("Then add the endpoint URL to your environment")
        logger.warning("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)