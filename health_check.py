import os
import sys

from prismalog.config import LoggingConfig
from prismalog.log import get_logger

from psy_supabase.config import TEXT_GENERATING_MODEL
from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.model_manager import get_embedding_provider
from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.common import is_github_actions

config_path = os.path.join(os.path.dirname(__file__), "config_logging.yaml")
LoggingConfig.initialize(config_file=config_path)

logger = get_logger(__name__)
if not is_github_actions():
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file
    logger.info("Local development: Loading environment from .env file")
else:
    logger.info("CI environment: Using GitHub secrets")


def main() -> bool:
    """Run a health check on the main components."""
    try:
        logger.info("Starting health check...")

        # Initialize with minimal dependencies for testing
        logger.info("Initializing text generator...")
        generator = TextGenerator(
            model_name=os.getenv("MODEL_NAME", TEXT_GENERATING_MODEL), device=os.getenv("DEVICE", "cpu")
        )
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and key must be set in environment variables.")
        logger.info("Initializing database manager...")
        db_manager = DatabaseManager(supabase_key=supabase_key, supabase_url=supabase_url, user_id="new_user_alex")
        logger.info("Database manager initialized successfully.")

        logger.info("Creating user schema...")
        db_manager.create_user_schema_sync()
        logger.info("User schema created successfully.")

        logger.info("Initializing RAG processor...")
        processor = RAGProcessor(db_manager=db_manager, generator=generator)
        logger.info("RAG processor initialized successfully.")

        logger.info("Testing RAG processor with sample question...")
        test_question = "How can I manage everyday anxiety?"

        embedding_provider = get_embedding_provider()
        test_embedding = embedding_provider.generate_embedding(test_question)
        logger.info("Generated test embedding with length %d", len(test_embedding))

        # Then use generate_response with a session ID that clearly identifies it as a session
        response = processor.generate_response(test_question, session_id="health_check")

        logger.info("Sample response: %s...", response[:100])

        logger.info("Cleaning up")
        cleanup_query = (
            """
        DROP SCHEMA IF EXISTS "%s" CASCADE;
        """
            % db_manager.schema_name
        )
        db_manager.supabase.rpc("sql", {"command": cleanup_query}).execute()
        logger.info("Schema %s dropped", db_manager.schema_name)
        logger.info("Health check completed successfully!")
        return True
    except Exception as e:
        logger.error("Health check failed: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
