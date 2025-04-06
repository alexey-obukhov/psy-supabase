import os
import sys
import uuid
import time
from datetime import datetime, timezone

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from psy_supabase.core.database import DatabaseManager
from school_logging.log import ColoredLogger

# Set up logger
logger = ColoredLogger(__name__)

# Load environment variables for Supabase credentials
from dotenv import load_dotenv
load_dotenv()

supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

class TestDatabaseFunctions:
    """Test class for database functions, particularly those involving session_id."""

    def setup_method(self):
        """Set up test environment."""
        try:
            # Create unique test user ID and session ID
            self.test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
            self.test_session_id = f"test_session_{uuid.uuid4().hex[:8]}"

            logger.info("Test user ID: %s", self.test_user_id)
            logger.info("Test session ID: %s", self.test_session_id)

            # Initialize database manager with test user
            self.db_manager = DatabaseManager(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                user_id=self.test_user_id
            )

            # Ensure schema exists
            logger.info("Creating schema and tables")
            schema_created = self.db_manager.create_user_schema_sync()
            if not schema_created:
                logger.error("Failed to create schema")

            # Verify schema exists
            check_query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.schemata
                WHERE schema_name = '{self.db_manager.schema_name}'
            );
            """
            check_response = self.db_manager.supabase.rpc('sql', {'command': check_query}).execute()
            logger.info("Schema exists check: %s", check_response.data)

            # Add a small delay to ensure schema creation completes
            time.sleep(2)

            # Ensure vector indexes
            logger.info("Creating vector indexes")
            self.db_manager.ensure_vector_indexes(None)

            logger.info("Setup complete")
        except Exception as e:
            logger.error("Setup error: %s", e)
            import traceback
            logger.error(traceback.format_exc())
            raise

    def test_save_and_retrieve_with_session_id(self):
        """
        Test saving and retrieving an interaction with a session ID as a complete workflow.

        This test verifies the critical end-to-end flow of:
        1. Saving an interaction with a session_id to the database
        2. Retrieving that interaction using the same session_id

        IMPORTANT: This must remain a single test rather than separate save/retrieve tests because:
        - It tests the complete workflow that real users experience
        - It verifies that data consistency guarantees between write and read operations work correctly
        - It ensures the session_id indexing functions properly for retrieval
        - It validates the eventual consistency model of the database system

        The short delay between save and retrieve operations accommodates the asynchronous
        nature of Supabase's storage layer. In production systems, you might implement a
        retry pattern with exponential backoff instead of a fixed sleep time.

        Failure modes that this test catches:
        - Session ID not being properly stored or indexed
        - Problems with the session_id column not being created
        - Incorrect metadata handling related to session_id
        - Data integrity issues between save and retrieve operations
        """        # Log for debugging
        logger.info("Ensuring tables exist before saving interaction")
        self.db_manager.ensure_schema_exists()

        # Generate a unique session ID
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        logger.info("Attempting to save interaction with session_id: %s", session_id)

        # Add an interaction with this session ID
        result = self.db_manager.add_interaction({
            'question': "This is a test question for retrieving by session ID",
            'answer': "This is a test answer for retrieving by session ID",
            'context': "Test context",
            'metadata': {
                'test_type': 'session_retrieval',
                'timestamp': datetime.now().isoformat()
            }
        }, session_id=session_id)

        # Verify save operation succeeded
        assert result, f"Failed to save interaction with session ID {session_id}"

        # Retrieve conversation history
        time.sleep(2)  # Wait for processing
        logger.info("Retrieving conversation history for session: %s", session_id)
        history = self.db_manager.get_conversation_history(session_id)

        # Now assert on the result
        assert len(history) > 0, "No history retrieved for the test session ID"

        # Verify the saved question is found
        found = False
        for item in history:
            if "test question for retrieving by session ID" in item.get('question', ''):
                found = True
                break

        assert found, "Saved interaction not found in retrieved history"

    def test_find_similar_documents_with_session_id(self):
        """Test finding similar documents with session_id."""
        # Create a few interactions with embeddings
        for i in range(3):
            question = f"This is test question {i} about vector search"
            answer = f"This is test answer {i} about vector search"
            context = "Testing vector search"
            metadata = {"topic": "vector_search", "subtopic": f"test_{i}"}

            # Save each interaction
            self.db_manager.save_interaction(
                context=context,
                question=question,
                answer=answer,
                metadata=metadata,
                session_id=self.test_session_id
            )

        # Wait for embeddings to be processed
        time.sleep(3)

        # Create an embedding for searching
        test_query = "Tell me about vector searching"
        embedding = self.db_manager.create_embedding(test_query)

        # Search with session_id
        results = self.db_manager.find_similar_documents_via_rpc(
            embedding=embedding,
            session_id=self.test_session_id,
            limit=5,
            similarity_threshold=0.1,  # Lower threshold for test
        )

        # Verify results
        assert isinstance(results, list), "Results should be a list"

        if len(results) > 0:
            # Check that results contain metadata
            for result in results:
                assert 'metadata' in result, "Result is missing metadata field"
                assert isinstance(result['metadata'], dict), "Metadata should be a dictionary"
                assert result['metadata'].get('session_id') == self.test_session_id, "Session ID mismatch in results"

        logger.info("Found %s similar documents with session ID", len(results))

    def test_dynamic_rag_with_session_id(self):
        """Test DynamicRAGRetriever with session_id."""
        from psy_supabase.core.dynamic_rag import DynamicRAGRetriever

        # Create a RAG retriever with the test session
        rag_retriever = DynamicRAGRetriever(
            db_manager=self.db_manager,
            session_id=self.test_session_id
        )

        # Add some test interactions
        questions = [
            "I'm feeling anxious about my upcoming presentation",
            "My anxiety seems to get worse when I have to speak in public",
            "I've been practicing deep breathing to manage anxiety"
        ]

        answers = [
            "It's normal to feel anxious about presentations. Have you tried any relaxation techniques?",
            "Public speaking anxiety is very common. Can you tell me more about what specifically worries you?",
            "Deep breathing is an excellent technique. How has it been working for you?"
        ]

        for q, a in zip(questions, answers):
            self.db_manager.save_interaction(
                context="Anxiety discussion",
                question=q,
                answer=a,
                metadata={"topic": "anxiety"},
                session_id=self.test_session_id
            )

        # Wait for processing
        time.sleep(3)

        # Test knowledge retrieval with the session context
        query = "What helps with anxiety?"
        result = rag_retriever.get_knowledge_by_query(query, limit=2)
        logger.info("Result: %s", result)
        assert isinstance(result, str), "Result should be a string"
        logger.info("RAG retrieval result: %s", result)

        # Test retrieving past interactions
        past = rag_retriever.get_past_interactions(session_id=self.test_session_id, limit=2)
        assert isinstance(past, list), "Past interactions should be a list of dictionaries"
        assert len(past) > 0, "No past interactions retrieved"

        logger.info("Past interactions: %s", past)

    def teardown_method(self):
        """Clean up after tests."""
        logger.info("Cleaning up test data for user %s", self.test_user_id)
        # Optional: Delete test schema to clean up
        try:
            self.db_manager.supabase.rpc('sql', {
                'command': f'DROP SCHEMA IF EXISTS "{self.db_manager.schema_name}" CASCADE;'
            }).execute()
            logger.info("Dropped test schema %s", self.db_manager.schema_name)
        except Exception as e:
            logger.error("Error cleaning up: %s", e)


def run_tests():
    """Run the test suite."""
    test = TestDatabaseFunctions()

    try:
        print("Setting up test environment...")
        test.setup_method()

        print("\nRunning test_save_and_retrieve_with_session_id...")
        test.test_save_and_retrieve_with_session_id()
        print("✓ Test passed!")

        print("\nRunning test_find_similar_documents_with_session_id...")
        test.test_find_similar_documents_with_session_id()
        print("✓ Test passed!")

        print("\nRunning test_dynamic_rag_with_session_id...")
        test.test_dynamic_rag_with_session_id()
        print("✓ Test passed!")

    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        test.teardown_method()
        print("Tests completed.")


if __name__ == "__main__":
    # Run the tests directly when this file is executed
    run_tests()