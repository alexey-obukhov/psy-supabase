import os
import time
import uuid

from dotenv import load_dotenv
from prismalog.config import LoggingConfig
from prismalog.log import get_logger

from psy_supabase.core.database import DatabaseManager
from psy_supabase.utilities.common import is_github_actions

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
LoggingConfig.initialize(config_file=config_path)

logger = get_logger(__name__)

if not is_github_actions():
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file
    logger.info("Local development: Loading environment from .env file")
else:
    logger.info("CI environment: Using GitHub secrets")


def main() -> None:
    """Debug database issues with session_id."""
    # Create test IDs
    test_user_id = f"test_debug_{uuid.uuid4().hex[:8]}"
    test_session_id = f"debug_session_{uuid.uuid4().hex[:8]}"

    logger.info("Test user ID: %s", test_user_id)
    logger.info("Test session ID: %s", test_session_id)

    # Initialize DatabaseManager
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    assert supabase_url is not None, "Supabase URL must be set"
    assert supabase_key is not None, "Supabase Key must be set"

    db_manager = DatabaseManager(supabase_url=supabase_url, supabase_key=supabase_key, user_id=test_user_id)

    # Step 1: Create schema and verify
    logger.info("Step 1: Creating schema and verifying")
    db_manager.create_user_schema_sync()

    # Verify schema exists
    check_query = (
        """
    SELECT EXISTS (
        SELECT FROM information_schema.schemata
        WHERE schema_name = '%s'
    );
    """
        % db_manager.schema_name
    )
    check_response = db_manager.supabase.rpc("sql", {"command": check_query}).execute()
    logger.info("Schema exists: %s", check_response.data)

    # Step 2: Verify tables
    logger.info("Step 2: Checking tables")
    tables_query = (
        """
    SELECT string_agg(table_name, ', ')
    FROM information_schema.tables
    WHERE table_schema = '%s'
    AND table_type = 'BASE TABLE';
    """
        % db_manager.schema_name
    )
    tables_response = db_manager.supabase.rpc("sql", {"command": tables_query}).execute()
    logger.info("Tables in schema %s: %s", db_manager.schema_name, tables_response.data)

    # Step 3: Check columns in interactions table
    logger.info("Step 3: Checking table structure")
    columns_query = (
        """
    SELECT string_agg(column_name, ', ')
    FROM information_schema.columns
    WHERE table_schema = '%s'
    AND table_name = 'interactions';
    """
        % db_manager.schema_name
    )
    columns_response = db_manager.supabase.rpc("sql", {"command": columns_query}).execute()
    logger.info("Columns in %s.interactions: %s", db_manager.schema_name, columns_response.data)

    # If session_id column is missing, add it
    if "session_id" not in str(columns_response.data).lower():
        logger.warning("session_id column missing, adding it")
        add_column_query = """
        ALTER TABLE "%s".interactions
        ADD COLUMN session_id TEXT;

        CREATE INDEX IF NOT EXISTS idx_%s_session_id
        ON "%s".interactions(session_id);
        """ % (
            db_manager.schema_name,
            db_manager.schema_name,
            db_manager.schema_name,
        )
        db_manager.supabase.rpc("sql", {"command": add_column_query}).execute()

    # Step 4: Direct SQL insertion to verify column
    logger.info("Step 4: Adding test interaction directly")
    logger.info("Inserting interaction with session_id %s", test_session_id)

    # Try to insert with both styles of column names
    insert_query = """
    INSERT INTO "%s".interactions
    (context, question, answer, metadata, session_id)
    VALUES
    ('Debug context', 'Debug question?', 'Debug answer',
     '{"topic": "debug", "session_id": "%s"}',
     '%s')
    RETURNING "interaction_id";
    """ % (
        db_manager.schema_name,
        test_session_id,
        test_session_id,
    )

    try:
        insert_response = db_manager.supabase.rpc("sql", {"command": insert_query}).execute()
        logger.info("Insert response: %s", insert_response.data)
    except Exception as e:
        logger.info("Insert response: %s", str(e))

        try:
            insert_response = db_manager.supabase.rpc("sql", {"command": insert_query}).execute()
            logger.info("Insert with lowercase: %s", insert_response.data)
        except Exception as e:
            logger.info("Insert with lowercase failed: %s", str(e))

    logger.info("Inserted interaction with ID: %s", insert_response.data)

    # Step 5: Retrieve conversation history
    logger.info("Step 5: Retrieving conversation history")
    time.sleep(1)  # Wait a moment for any async processes

    logger.info("Getting conversation history for session_id %s", test_session_id)
    history_query = """
    SELECT * FROM "%s".interactions
    WHERE session_id = '%s'
    OR metadata->>'session_id' = '%s';
    """ % (
        db_manager.schema_name,
        test_session_id,
        test_session_id,
    )

    try:
        history_response = db_manager.supabase.rpc("sql", {"command": history_query}).execute()
        logger.info("Query response: %s", history_response.data)
    except Exception as e:
        logger.warning("Query response: %s", str(e))

    # Step 6: Test DatabaseManager.save_interaction
    logger.info("Step 6: Testing DatabaseManager.save_interaction method")
    result = db_manager.save_interaction(
        context="Debug context via method",
        question="Debug question via method?",
        answer="Debug answer via method",
        metadata={"topic": "debug", "source": "method"},
        session_id=test_session_id,
    )

    logger.info("save_interaction result: %s", result)

    # Step 7: Check conversation history after save_interaction
    logger.info("Step 7: Checking history after using save_interaction")
    time.sleep(1)

    logger.info("Getting conversation history for session_id %s", test_session_id)
    try:
        history = db_manager.get_conversation_history(test_session_id)
        logger.info("History items: %d", len(history))
        if history:
            logger.info("First item: %s", history[0])
    except Exception as e:
        logger.info("Query response: %s", str(e))

    # Step 8: Clean up
    logger.info("Step 8: Cleaning up")
    cleanup_query = (
        """
    DROP SCHEMA IF EXISTS "%s" CASCADE;
    """
        % db_manager.schema_name
    )
    db_manager.supabase.rpc("sql", {"command": cleanup_query}).execute()
    logger.info("Schema %s dropped", db_manager.schema_name)


if __name__ == "__main__":
    main()
