import uuid
from unittest import TestCase
from school_logging.log import ColoredLogger

class DatabaseTestBase(TestCase):
    """Base class for tests that need database access and diagnostics."""

    def setUp(self):
        """Common test setup."""
        self.logger = ColoredLogger(__name__)
        self.logger.info("Setting up database test environment")
        self.test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        self.test_session_id = f"test_session_{uuid.uuid4().hex[:10]}"
        self.logger.info("Test user ID: %s", self.test_user_id)
        self.logger.info("Test session ID: %s", self.test_session_id)

    def setup_test_environment(self):
        """Set up the test environment, creating necessary tables."""
        # Code copied from _setup_test_environment in PainPointDetectionTester

    def diagnose_database_issues(self):
        """Diagnose common database issues affecting tests."""
        self.logger.info("=== Running Database Diagnostics ===")

        # Check if db_manager exists (could be mock or real)
        if not hasattr(self, 'db_manager') and not hasattr(self, 'mock_db'):
            self.logger.error("No db_manager or mock_db found to perform diagnostics")
            raise ValueError("No db_manager or mock_db found to perform diagnostics")

        # Use either real or mock db_manager
        db = getattr(self, 'db_manager', getattr(self, 'mock_db', None))

        try:
            self.logger.info("Checking database structure and connectivity")

            if hasattr(db, 'supabase'):
                try:
                    # Execute a simple query to check connectivity and actually use the result
                    query_result = db.supabase.rpc('sql', {'command': "SELECT 'Connection test successful' as status;"}).execute()

                    # Extract and log the actual result data
                    if hasattr(query_result, 'data') and query_result.data:
                        status = query_result.data[0].get('status', 'Unknown')
                        self.logger.info("Database connection test result: %s", status)
                    else:
                        self.logger.warning("Database connection test returned no data")

                    # Check for the current schema
                    schema_query = "SELECT current_schema() as schema;"
                    schema_result = db.supabase.rpc('sql', {'command': schema_query}).execute()

                    if hasattr(schema_result, 'data') and schema_result.data:
                        current_schema = schema_result.data[0].get('schema', 'Unknown')
                        self.logger.info("Currently using schema: %s", current_schema)

                    # Check for tables in the current schema
                    tables_query = """
                    SELECT table_name, table_type
                    FROM information_schema.tables
                    WHERE table_schema = current_schema()
                    ORDER BY table_name;
                    """
                    tables_result = db.supabase.rpc('sql', {'command': tables_query}).execute()

                    if hasattr(tables_result, 'data') and tables_result.data:
                        self.logger.info("Found %d tables in schema %s:", len(tables_result.data), current_schema)
                        for table in tables_result.data:
                            self.logger.info("  - %s (%s)", table.get('table_name'), table.get('table_type'))
                    else:
                        self.logger.warning("No tables found in schema %s", current_schema)

                except Exception as e:
                    self.logger.error("Database connection failed: %s", e)
            else:
                self.logger.warning("Using mock database - connectivity tests skipped")

        except Exception as e:
            self.logger.error("Error during diagnostics: %s", e)
            import traceback
            self.logger.error(traceback.format_exc())

    def direct_table_check(self):
        """Run a direct SQL check to get table structure information"""
        # Code copied from direct_table_check in PainPointDetectionTester

    def run_diagnostics_summary(self):
        """Run a basic diagnostic summary that works for all database tests."""
        self.logger.info("=== Basic Database Diagnostics Summary ===")
        self.logger.info("Test session ID: %s", self.test_session_id)

        # Check if we have a db_manager or mock_db
        has_real_db = hasattr(self, 'db_manager')
        has_mock_db = hasattr(self, 'mock_db')

        if has_real_db:
            self.logger.info("Using real database connection")
            # Basic DB checks like schema existence

        elif has_mock_db:
            self.logger.info("Using mock database")
            # Basic mock checks
        else:
            self.logger.warning("No database manager found!")

        return has_real_db, has_mock_db  # Return info about what kind of DB we have

    def tearDown(self):
        """Common test teardown with basic diagnostics."""
        try:
            # Always run basic diagnostics
            self.run_diagnostics_summary()
        except Exception as e:
            self.logger.error("Error in base tearDown diagnostics: %s", e)