import os
import sys
import asyncio
import logging
import gc
import time
import threading
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
from dataclasses import asdict

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core logic
from core_logic.pinecone_vector import PineconeVectorClient
from core_logic.rag import RAGRetriever
from core_logic.conversational_agent import ConversationalAgent
from core_logic.knowledge_bridge import PharmaceuticalKnowledgeBridge, EnhancedRAGRetriever
from core_logic.model_manager import model_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")

app = Flask(__name__)

# Configure CORS
CORS(
    app,
    origins=[
        "https://pharma-assistant-55.web.app",
        "https://pharma-assistant-55.firebaseapp.com",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ],
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True,
)

# Global agent instance
agent = None
_init_lock = threading.Lock()
_agent_initialized = False

def build_agent_singleton():
    """Build the conversational agent with cached models"""
    global agent
    
    if agent is not None:
        return agent
    
    logger.info("Initializing ConversationalAgent singleton...")
    
    try:
        start_time = time.time()
        
        # Ensure models are loaded
        model_manager.preload_all()
        logger.info("Models ready, status: %s", model_manager.get_status())
        
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
        
        total_time = time.time() - start_time
        logger.info("ConversationalAgent singleton created successfully in %.2fs", total_time)
        
        # Force garbage collection
        gc.collect()
        
        return agent
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        raise

def ensure_agent_initialized():
    """Ensure agent is initialized (called on first request)"""
    global agent, _agent_initialized
    
    if _agent_initialized and agent is not None:
        return
    
    with _init_lock:
        if not _agent_initialized or agent is None:
            agent = build_agent_singleton()
            _agent_initialized = True

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Max-Age", "3600")
        return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy',
        'service': 'pharma-assistant',
        'timestamp': time.time(),
    }
    
    # Check if agent is initialized
    if agent is not None:
        health_status['agent_status'] = 'ready'
    else:
        health_status['agent_status'] = 'not_initialized'
    
    # Check model status
    health_status['models'] = model_manager.get_status()
    
    return jsonify(health_status), 200

@app.route('/api/warmup', methods=['GET'])
@app.route('/_ah/warmup', methods=['GET'])
def warmup():
    """Warmup endpoint for Cloud Run"""
    logger.info("Warmup request received")
    start_time = time.time()
    
    try:
        ensure_agent_initialized()
        
        # Do a test query to warm up the pipeline
        if agent:
            test_results = agent.retriever.base_retriever.vector_client.query(
                text="warmup",
                top_k=1
            )
            logger.info("Warmup test successful")
        
        total_time = time.time() - start_time
        return jsonify({
            'status': 'warm',
            'duration': total_time,
            'agent_ready': agent is not None
        }), 200
        
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Main chat endpoint"""
    ensure_agent_initialized()
    
    if agent is None:
        return jsonify({'error': 'Service is initializing, please try again'}), 503
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    user_query = data['text']
    conversation_history = data.get('history', [])
    
    logger.info(f"Received query: '{user_query[:100]}...'")
    
    try:
        # Create async event loop for the async agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_agent():
            return await agent.handle(
                user_query=user_query,
                conversation_history=conversation_history
            )
        
        # Run with timeout
        async def run_with_timeout():
            return await asyncio.wait_for(run_agent(), timeout=30.0)
        
        decision = loop.run_until_complete(run_with_timeout())
        loop.close()
        
        # Clean up memory
        gc.collect()
        
        # Return response
        return jsonify({
            'response_text': decision.response_text,
            'safety_labels': decision.safety_labels,
            'trace': decision.trace if data.get('include_trace', False) else None
        })
        
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        logger.error(f"Error handling chat: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the agent's cache"""
    if agent is None:
        return jsonify({'error': 'Agent not initialized'}), 503
    
    try:
        agent.clear_answer_cache()
        if hasattr(agent.retriever, 'clear_cache'):
            agent.retriever.clear_cache()
        
        gc.collect()
        
        logger.info("Cache cleared successfully")
        return jsonify({'success': True, 'message': 'Cache cleared'})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        return jsonify({'error': 'Failed to clear cache'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    
    # Check for required environment variables
    required_vars = ['PINECONE_API_KEY', 'ANTHROPIC_API_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
    
    app.run(host='0.0.0.0', port=port, debug=False)