import os
import sys
import time
import uuid
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Load environment variables
load_dotenv()
from school_logging.log import ColoredLogger
from psy_supabase.core.database import DatabaseManager

logger = ColoredLogger(__name__)


def main():
    """Debug and fix database issues with session_id."""
    # Create test IDs
    test_user_id = f"test_debug_{uuid.uuid4().hex[:8]}"
    test_session_id = f"debug_session_{uuid.uuid4().hex[:8]}"

    logger.info(f"Test user ID: {test_user_id}")
    logger.info(f"Test session ID: {test_session_id}")

    # Initialize DatabaseManager
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')

    db_manager = DatabaseManager(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        user_id=test_user_id
    )

    # Step 1: Create schema and verify
    logger.info("Step 1: Creating schema and verifying")
    db_manager.create_user_schema_sync()

    # Verify schema exists
    check_query = f"""
    SELECT EXISTS (
        SELECT FROM information_schema.schemata
        WHERE schema_name = '{db_manager.schema_name}'
    );
    """
    check_response = db_manager.supabase.rpc('sql', {'command': check_query}).execute()
    logger.info(f"Schema exists: {check_response.data}")

    # Step 2: Verify tables
    logger.info("Step 2: Checking tables")
    tables_query = f"""
    SELECT string_agg(table_name, ', ')
    FROM information_schema.tables
    WHERE table_schema = '{db_manager.schema_name}'
    AND table_type = 'BASE TABLE';
    """
    tables_response = db_manager.supabase.rpc('sql', {'command': tables_query}).execute()
    logger.info(f"Tables in schema {db_manager.schema_name}: {tables_response.data}")

    # Step 3: Check columns in interactions table
    logger.info("Step 3: Checking table structure")
    columns_query = f"""
    SELECT string_agg(column_name, ', ')
    FROM information_schema.columns
    WHERE table_schema = '{db_manager.schema_name}'
    AND table_name = 'interactions';
    """
    columns_response = db_manager.supabase.rpc('sql', {'command': columns_query}).execute()
    logger.info(f"Columns in {db_manager.schema_name}.interactions: {columns_response.data}")

    # If session_id column is missing, add it
    if 'session_id' not in str(columns_response.data).lower():
        logger.warning("session_id column missing, adding it")
        add_column_query = f"""
        ALTER TABLE "{db_manager.schema_name}".interactions
        ADD COLUMN session_id TEXT;

        CREATE INDEX IF NOT EXISTS idx_{db_manager.schema_name}_session_id
        ON "{db_manager.schema_name}".interactions(session_id);
        """
        db_manager.supabase.rpc('sql', {'command': add_column_query}).execute()

    # Step 4: Direct SQL insertion to verify column
    logger.info("Step 4: Adding test interaction directly")
    logger.info(f"Inserting interaction with session_id {test_session_id}")

    # Try to insert with both styles of column names
    insert_query = f"""
    INSERT INTO "{db_manager.schema_name}".interactions
    (context, question, answer, metadata, session_id)
    VALUES
    ('Debug context', 'Debug question?', 'Debug answer',
     '{{"topic": "debug", "session_id": "{test_session_id}"}}',
     '{test_session_id}')
    RETURNING "interactionID";
    """

    try:
        insert_response = db_manager.supabase.rpc('sql', {'command': insert_query}).execute()
        logger.info(f"Insert response: {insert_response.data}")
    except Exception as e:
        logger.info(f"Insert response: {str(e)}")

        # Try with lowercase interactionid
        insert_query = insert_query.replace('"interactionID"', '"interactionid"')
        try:
            insert_response = db_manager.supabase.rpc('sql', {'command': insert_query}).execute()
            logger.info(f"Insert with lowercase: {insert_response.data}")
        except Exception as e:
            logger.info(f"Insert with lowercase failed: {str(e)}")

    logger.info(f"Inserted interaction with ID: {insert_response.data}")

    # Step 5: Retrieve conversation history
    logger.info("Step 5: Retrieving conversation history")
    time.sleep(1)  # Wait a moment for any async processes

    logger.info(f"Getting conversation history for session_id {test_session_id}")
    history_query = f"""
    SELECT * FROM "{db_manager.schema_name}".interactions
    WHERE session_id = '{test_session_id}'
    OR metadata->>'session_id' = '{test_session_id}';
    """

    try:
        history_response = db_manager.supabase.rpc('sql', {'command': history_query}).execute()
        logger.info(f"Query response: {history_response.data}")
    except Exception as e:
        logger.warning(f"Query response: {str(e)}")

    # Step 6: Test DatabaseManager.save_interaction
    logger.info("Step 6: Testing DatabaseManager.save_interaction method")
    result = db_manager.save_interaction(
        context="Debug context via method",
        question="Debug question via method?",
        answer="Debug answer via method",
        metadata={"topic": "debug", "source": "method"},
        session_id=test_session_id
    )

    logger.info(f"save_interaction result: {result}")

    # Step 7: Check conversation history after save_interaction
    logger.info("Step 7: Checking history after using save_interaction")
    time.sleep(1)

    logger.info(f"Getting conversation history for session_id {test_session_id}")
    try:
        history = db_manager.get_conversation_history(test_session_id)
        logger.info(f"History items: {len(history)}")
        if history:
            logger.info(f"First item: {history[0]}")
    except Exception as e:
        logger.info(f"Query response: {str(e)}")

    # Step 8: Clean up
    logger.info("Step 8: Cleaning up")
    cleanup_query = f"""
    DROP SCHEMA IF EXISTS "{db_manager.schema_name}" CASCADE;
    """
    db_manager.supabase.rpc('sql', {'command': cleanup_query}).execute()
    logger.info(f"Schema {db_manager.schema_name} dropped")

if __name__ == "__main__":
    main()