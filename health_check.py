from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.core.database import DatabaseManager
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run a health check on the main components."""
    try:
        logger.info("Starting health check...")
        load_dotenv()  # Load environment variables

        # Initialize with minimal dependencies for testing
        logger.info("Initializing text generator...")
        generator = TextGenerator(
            model_name=os.getenv("MODEL_NAME", "microsoft/phi-1_5"),
            device=os.getenv("DEVICE", "cpu")
        )
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and key must be set in environment variables.")
        logger.info("Initializing database manager...")
        db_manager = DatabaseManager(supabase_key=supabase_key, supabase_url=supabase_url, user_id="new_user_alex")
        logger.info("Database manager initialized successfully.")
        logger.info("Initializing RAG processor...")
        processor = RAGProcessor(db_manager=db_manager, generator=generator)

        # Test basic functionality
        logger.info("Testing RAG processor with sample question...")
        test_question = "How can I manage everyday anxiety?"
        response = processor.generate_response(test_question, session_id="health_check")

        logger.info(f"Sample response: {response[:100]}...")
        logger.info("Health check completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)