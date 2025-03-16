import os
import sys
import logging
import subprocess
import time
import random
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify, g
from psy_supabase.utilities.text_utils import cleanup_memory
from school_logging.log import ColoredLogger
import spacy

from typeguard import install_import_hook
install_import_hook('psy_supabase')

# Configure logging first thing
from psy_supabase.utilities.logging_config import configure_logging
configure_logging(level=logging.INFO)  # Use logging.DEBUG for development

# Load environment variables (only once)
load_dotenv()

import torch
import multiprocessing as mp

# Set up logging
logger = ColoredLogger(__name__)

# Memory management variables
last_memory_cleanup = time.time()
request_counter = 0
CLEANUP_THRESHOLD = 10  # Clean up after 10 requests
CLEANUP_TIME_THRESHOLD = 300  # Clean up after 5 minutes

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
from psy_supabase.core.model_manager import get_model_manager

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

# Define model name at module level for consistency
model_name = "microsoft/phi-1_5"

app = Flask(__name__)

def initialize_app():
    """Set up the application before the first request."""
    logger.info("Setting up application...")
    logger.info(f"Welcome to the Therapy AI Assistant! Using model: {model_name} on {device}")
    
    # Initialize the model manager but don't load the model yet
    # This just sets up the instance which will lazy-load when needed
    get_model_manager(model_name, device)
    logger.info(f"Model manager initialized for {model_name}")

# Call initialize directly
initialize_app()

def should_cleanup_memory():
    """Determine if we should clean up GPU memory based on request count and time."""
    global request_counter, last_memory_cleanup
    
    request_counter += 1
    current_time = time.time()
    time_since_cleanup = current_time - last_memory_cleanup
    
    # Clean up if:
    # 1. We've processed enough requests OR
    # 2. It's been long enough since last cleanup OR
    # 3. Randomly with low probability (to avoid memory fragmentation)
    if (request_counter >= CLEANUP_THRESHOLD or 
        time_since_cleanup >= CLEANUP_TIME_THRESHOLD or 
        random.random() < 0.05):  # 5% chance to clean up
        
        request_counter = 0
        last_memory_cleanup = current_time
        return True
    
    return False

@app.teardown_request
def teardown_request(exception=None):
    """Clean up after request if needed."""
    if device == "cuda" and should_cleanup_memory():
        cleanup_memory()

@app.before_request
def before_request():
    """Initialize DatabaseManager before each request."""
    # Skip for preflight requests
    if request.method == 'OPTIONS':
        return

    # Skip for paths that don't need authentication
    if request.path in ['/health', '/memory_status', '/free_memory']:
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

@app.route('/memory_status', methods=['GET'])
def memory_status():
    """Report memory usage status (useful for monitoring)."""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        reserved = torch.cuda.memory_reserved(0) / 1e9  # GB
        allocated = torch.cuda.memory_allocated(0) / 1e9  # GB
        free = total - reserved
        
        return jsonify({
            'device': torch.cuda.get_device_name(0),
            'total_memory_gb': round(total, 2),
            'reserved_memory_gb': round(reserved, 2),
            'allocated_memory_gb': round(allocated, 2),
            'free_memory_gb': round(free, 2),
            'utilization_percent': round((reserved / total) * 100, 2)
        })
    else:
        return jsonify({'device': 'CPU', 'message': 'No CUDA device available'})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({"error": "Missing question parameter"}), 400
            
        user_id = request.headers.get('X-User-ID', 'default_user')
        question = data['question']
        
        # Log the incoming request
        logger.info(f"Received chat request from user {user_id}: {question[:50]}...")

        # Check if knowledge base is empty and initialize if needed
        # This should be done before expensive model operations
        try:
            # Use a SQL query through RPC instead of direct table access
            # This avoids errors if the table doesn't exist
            check_query = f"""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = '{g.db_manager.schema_name}' 
            AND table_name = 'knowledge_base';
            """
            
            table_exists = g.db_manager.supabase.rpc('sql', {'command': check_query}).execute()
            
            if table_exists.data and table_exists.data[0] == '0':
                logger.info(f"Knowledge base table doesn't exist for {user_id}, creating it...")
                g.db_manager.create_user_schema_sync()
            
            # Now check if the table has data
            count_query = f"""
            SELECT COUNT(*) FROM "{g.db_manager.schema_name}".knowledge_base;
            """
            
            try:
                count_result = g.db_manager.supabase.rpc('sql', {'command': count_query}).execute()
                if count_result.data and count_result.data[0] == '0':
                    logger.info(f"Knowledge base for {user_id} is empty, initializing...")
                    g.db_manager.initialize_knowledge_base(user_id)
            except Exception as count_e:
                # If this fails, the table might not exist despite our earlier check
                logger.error(f"Error checking knowledge base count: {count_e}")
                g.db_manager.initialize_knowledge_base(user_id)
                
        except Exception as kb_e:
            logger.error(f"Error checking or initializing knowledge base: {kb_e}")
            # Continue with chat process even if this fails

        # Get the model manager instance and then get the generator
        model_manager = get_model_manager(model_name, device)
        generator = model_manager.get_generator()
        
        # Create a RAG processor using the retrieved documents
        rag_processor = RAGProcessor(g.db_manager, generator, intelligent_processing_enabled)

        try:
            # Generate response
            response = rag_processor.generate_response(user_question=question,
                                                       session_id=user_id,
                                                       device=device,
                                                       question_id=0,)
            
            # Validate response before returning
            if not response or len(response.strip()) < 10:
                logger.error(f"Invalid response generated: {response}")
                response = "I understand you're having difficulty with asking questions. Would you like to explore what makes this challenging for you? I'm here to support you."
            
            # Additional safety check for inappropriate response patterns
            if any(pattern in response for pattern in ["# YOUR CODE HERE", "SOLUTION:", "Answer the following:", "# 1.", "# 2."]):
                logger.error(f"Code template detected in response: {response}")
                response = "I notice you're concerned about asking questions and feeling stuck. Many people find this challenging. Would you like to explore what might help you feel more comfortable asking questions?"
        finally:
            # Always ensure we free memory for expensive operations
            # Chat is the most memory-intensive operation, so we clean up explicitly
            if device == "cuda":
                cleanup_memory()
        
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"response": "I apologize, but I encountered an error. Could you try expressing your concern in a different way?"}), 500

@app.route('/add_document', methods=['POST'])
def add_document():
    """Handles document addition requests."""
    try:
        data = request.get_json()
        content = data.get('content')

        if not content:
            return jsonify({'error': 'Missing content'}), 400

        # Get model manager instance
        model_manager = get_model_manager(model_name, device)
        
        try:
            # Use the embedding method directly from model manager
            embedding = model_manager.generate_embedding(content)
            
            if embedding is not None:
                # Store document with embedding
                g.db_manager.add_document_to_knowledge_base(content, embedding)
                return jsonify({'message': 'Document added successfully'})
            else:
                return jsonify({'error': 'Failed to generate embedding'}), 500
        finally:
            # Clean up if this was an expensive operation (long content)
            if len(content) > 5000 and device == "cuda":
                cleanup_memory()
    except Exception as e:
        logger.error("Error in add_document endpoint: %s", e)
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/get_documents', methods=['GET'])
def get_documents():
    """Retrieves all documents for the authenticated user."""
    documents = g.db_manager.get_all_documents_and_embeddings()
    return jsonify({'documents': documents})

@app.route('/optimize_vectors', methods=['POST'])
def optimize_vectors():
    """Optimize vector operations for the authenticated user."""
    try:
        # Get model manager for embedding generation
        model_manager = get_model_manager(model_name, device)
        
        # Use the database manager to optimize vector operations
        # First ensure all interactions have embedding column
        g.db_manager.add_embedding_column_to_interactions()
        
        # Find interactions without embeddings
        interactions = g.db_manager.get_interactions_without_embeddings()
        
        if not interactions:
            return jsonify({
                'message': 'No interactions found that need embeddings.'
            })
        
        total_interactions = len(interactions)
        batch_size = min(10, total_interactions)  # Process in smaller batches
        enriched_count = 0
        
        logger.info(f"Starting vector optimization: {total_interactions} interactions to process")
        
        try:
            # Process in batches to avoid memory issues
            for i in range(0, total_interactions, batch_size):
                batch = interactions[i:i+batch_size]
                
                for interaction in batch:
                    try:
                        interaction_id = interaction.get('interactionid')
                        question = interaction.get('question', '')
                        answer = interaction.get('answer', '')
                        
                        # Generate embedding from combined text
                        combined_text = f"Question: {question}\nAnswer: {answer}"
                        embedding = model_manager.generate_embedding(combined_text)
                        
                        if embedding:
                            # Add embedding to interaction
                            if g.db_manager.add_embedding_to_interaction(interaction_id, embedding):
                                enriched_count += 1
                    except Exception as e:
                        logger.error(f"Error enriching interaction: {e}")
                        continue
                
                # After each batch, clean up memory
                if device == "cuda":
                    cleanup_memory()
                
                # Log progress
                logger.info(f"Processed {min(i + batch_size, total_interactions)}/{total_interactions} interactions")
        finally:
            # Ensure memory is cleaned up after the operation
            if device == "cuda":
                cleanup_memory()
            
        # Ensure vector indexes exist
        g.db_manager.ensure_vector_indexes()
        
        return jsonify({
            'message': f'Vector operations optimized. {enriched_count}/{total_interactions} interactions enriched with embeddings.'
        })
    except Exception as e:
        logger.error(f"Error in optimize_vectors endpoint: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/free_memory', methods=['POST'])
def free_memory():
    """Explicitly free GPU memory on demand."""
    if device != "cuda":
        return jsonify({'message': 'Running on CPU, no GPU memory to free'})
    
    try:
        cleanup_memory()
        return jsonify({'message': 'GPU memory freed successfully'})
    except Exception as e:
        logger.error(f"Error freeing memory: {e}")
        return jsonify({'error': f'Failed to free memory: {str(e)}'}), 500

if __name__ == '__main__':
    logger = ColoredLogger("psy_supabase")
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    app.run(debug=False, port=5008)
