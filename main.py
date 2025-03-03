import os
from dotenv import load_dotenv
import torch
import traceback
import multiprocessing as mp
from flask import Flask, request, jsonify, g
import logging
from supabase import create_client, Client

from rag_processor import RAGProcessor
from database import DatabaseManager
from model_manager import ModelManager

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Disable tokenizer parallelism ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

@app.before_first_request
def initialize_app():
    """Set up the application before the first request."""
    logger.info("Setting up application...")
    # Any global initialization can go here

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


if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    load_dotenv()

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    intelligent_processing_enabled = os.environ.get("INTELLIGENT_PROCESSING_ENABLED", None)

    if not supabase_url or not supabase_key:
        logger.critical("Error: Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
        exit()

    # --- Use GPU if available ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Replace direct TextGenerator initialization with ModelManager
    model_manager = ModelManager("microsoft/phi-1_5", device)

    logger.info("Welcome to the Therapy AI Assistant!")

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

    app.run(debug=False, port=5008)


class DatabaseManager:
    def __init__(self, supabase_url: str, supabase_key: str, user_id: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.user_id = user_id  # Store the user_id for schema creation
        self.schema_name = f"user_{user_id}"  # Dynamically generate the schema name based on user_id
        # Initialize the Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        logger.debug(f"DatabaseManager initialized for user: {self.user_id}")
