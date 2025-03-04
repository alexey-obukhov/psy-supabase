import os
import sys
import logging
import subprocess
from dotenv import load_dotenv
from flask import Flask, request, jsonify, g
import spacy

# Configure logging first thing
from psy_supabase.utilities.logging_config import configure_logging
configure_logging(level=logging.INFO)  # Use logging.DEBUG for development

# Load environment variables
load_dotenv()

import torch
import multiprocessing as mp

# Load environment variables at module level
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def ensure_spacy_model():
    """Ensure that the spaCy model is available, downloading if necessary."""
    try:
        # Try to load the spaCy model
        spacy.load("en_core_web_sm")
        logger.info("Successfully loaded spaCy model 'en_core_web_sm'")
    except OSError:
        # Model not found, attempt to download
        logger.warning("SpaCy model 'en_core_web_sm' not found. Attempting to download...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            logger.info("Successfully downloaded spaCy model 'en_core_web_sm'")
        except Exception as e:
            logger.error(f"Failed to download spaCy model: {e}")
            logger.error("Please install it manually with: python -m spacy download en_core_web_sm")
            # Don't raise an exception, application might still work without the model

# Ensure spaCy model is available at startup
ensure_spacy_model()

from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.model_manager import ModelManager

# --- Disable tokenizer parallelism ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Move these variables to module level so they're available regardless of how the app runs
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
intelligent_processing_enabled = os.environ.get("INTELLIGENT_PROCESS_ENABLED", True)

# Validate environment variables at module level
if not supabase_url or not supabase_key:
    logger.critical("Error: Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
    # Don't exit here, as it would prevent module import

# --- Use GPU if available ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model manager at module level
model_manager = ModelManager("microsoft/phi-1_5", device)

app = Flask(__name__)

def initialize_app():
    """Set up the application before the first request."""
    logger.info("Setting up application...")
    logger.info("Welcome to the Therapy AI Assistant!")

# Call initialize directly
initialize_app()

@app.before_request
def before_request():
    """Initialize DatabaseManager before each request."""
    # Skip for preflight requests
    if request.method == 'OPTIONS':
        return

    # Skip for paths that don't need authentication
    if request.path in ['/health', '/status']:
        return
        
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({'error': 'User not authenticated'}), 401

    # Store user_id in Flask's 'g' object
    g.user_id = user_id

    # Initialize DatabaseManager and store in 'g'
    g.db_manager = DatabaseManager(supabase_url, supabase_key, g.user_id)

    # Create user schema synchronously
    schema_created = g.db_manager.create_user_schema_sync()
    
    if not schema_created:
        return jsonify({'error': 'Failed to create user schema'}), 500
        
    # Add vector index to knowledge base (new line)
    g.db_manager.add_vector_index_to_knowledge_base()

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({'status': 'ok'})

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat interactions."""
    try:
        data = request.json
        user_question = data.get('question')
        user_id = request.headers.get('X-User-ID')
        session_id = f"user_{user_id}"
        
        # Get the generator only when needed
        generator = model_manager.get_generator()
        
        # Create a RAG processor using the retrieved documents
        # This leverages Supabase embeddings more efficiently
        rag_processor = RAGProcessor(g.db_manager, generator, intelligent_processing_enabled)
        
        # Generate response
        response = rag_processor.generate_response(
            user_question, 
            device, 
            session_id=session_id
        )
        
        # Free GPU memory immediately after use
        model_manager.free_memory()
        
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        # Always free memory, even on error
        model_manager.free_memory()
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.route('/add_document', methods=['POST'])
def add_document():
    """Handles document addition requests."""
    try:
        data = request.get_json()
        content = data.get('content')

        if not content:
            return jsonify({'error': 'Missing content'}), 400

        # Get the generator from model manager
        generator = model_manager.get_generator()
        
        # Generate embedding and store the document
        embedding = generator.get_embedding(content)
        if embedding is not None:
            g.db_manager.add_document_to_knowledge_base(content, embedding.tolist())
            # Free GPU memory after use
            model_manager.free_memory()
            return jsonify({'message': 'Document added successfully'})
        else:
            # Free GPU memory even if embedding failed
            model_manager.free_memory()
            return jsonify({'error': 'Failed to generate embedding'}), 500
    except Exception as e:
        logger.error("Error in add_document endpoint: %s", e)
        # Always free memory on error
        model_manager.free_memory()
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/get_documents', methods=['GET'])
def get_documents():
    """Retrieves all documents for the authenticated user."""
    documents = g.db_manager.get_all_documents_and_embeddings()
    return jsonify({'documents': documents})

if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    app.run(debug=False, port=5008)
