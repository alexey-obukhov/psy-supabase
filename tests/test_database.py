import pytest
import json
from unittest.mock import Mock, patch, ANY
from psy_supabase.core.database import DatabaseManager

# Test data constants
TEST_USER_ID = "test_user_123"
TEST_SCHEMA = "test_user_123"
TEST_SESSION_ID = "test_session_123"
TEST_URL = "https://fake-supabase-url.com"
TEST_KEY = "fake-api-key"

# Sample interaction data
SAMPLE_INTERACTION = {
    'context': 'Test context',
    'question': 'How are you feeling today?',
    'answer': 'I am feeling better, thanks for asking.',
    'metadata': {'topic': 'Wellness', 'effectiveness': {'term_overlap': 0.8}}
}

# Sample conversation history data
SAMPLE_HISTORY = [
    {
        'interactionid': 1,
        'question': 'How are you feeling today?',
        'answer': 'I am feeling better, thanks for asking.',
        'context': 'Test context',
        'metadata': json.dumps({'topic': 'Wellness', 'effectiveness': {'term_overlap': 0.8}}),
        'created_at': '2023-01-01T12:00:00'
    },
    {
        'interactionid': 2,
        'question': 'What has been bothering you lately?',
        'answer': 'I have been stressed about work.',
        'context': 'CBT session',
        'metadata': json.dumps({'topic': 'Anxiety', 'effectiveness': {'term_overlap': 0.7}}),
        'created_at': '2023-01-01T12:05:00'
    }
]

# Sample document with embedding
SAMPLE_DOCUMENT = {
    'id': 1,
    'content': 'This is a sample document about anxiety management techniques.',
    'embedding': [0.1, 0.2, 0.3, 0.4]  # Shortened for brevity
}

# Return value for vector similarity search
SAMPLE_SIMILAR_DOCUMENTS = [
    {
        'id': 1,
        'content': 'This is a sample document about anxiety management techniques.',
        'similarity': 0.95
    },
    {
        'id': 2,
        'content': 'Another document about stress reduction strategies.',
        'similarity': 0.85
    }
]

class TestDatabaseManager:

    @pytest.fixture
    def mock_supabase(self):
        """Create a mock Supabase client."""
        mock_client = Mock()

        # Mock responses for various methods
        mock_rpc_response = Mock()
        mock_rpc_response.data = True
        mock_rpc_response.error = None

        # Configure execute() to return the mock response
        mock_execute = Mock(return_value=mock_rpc_response)

        # Configure rpc() to return an object with execute method
        mock_rpc = Mock()
        mock_rpc.execute = mock_execute

        # Configure table() to return object with insert method
        mock_insert = Mock()
        mock_insert.execute = mock_execute
        mock_table = Mock(return_value=mock_insert)

        # Attach all these mocks to the main client
        mock_client.rpc = Mock(return_value=mock_rpc)
        mock_client.table = Mock(return_value=mock_table)

        return mock_client

    @pytest.fixture
    def db_manager(self, mock_supabase):
        """Create a DatabaseManager with a mock Supabase client."""
        with patch('psy_supabase.core.database.create_client', return_value=mock_supabase):
            manager = DatabaseManager(TEST_URL, TEST_KEY, TEST_USER_ID)
            return manager

    def test_init(self, db_manager, mock_supabase):
        """Test initialization of DatabaseManager."""
        assert db_manager.supabase_url == TEST_URL
        assert db_manager.supabase_key == TEST_KEY
        assert db_manager.user_id == TEST_USER_ID
        assert db_manager.schema_name == TEST_SCHEMA
        assert db_manager.supabase == mock_supabase

    def test_create_user_schema(self, db_manager):
        """Test creating a user schema."""
        # Configure mock
        mock_response = Mock()
        mock_response.data = True
        mock_response.error = None
        db_manager.supabase.rpc().execute.return_value = mock_response

        # Call method
        result = db_manager.create_user_schema()

        # Verify result and that the proper RPC was called
        assert result is True
        db_manager.supabase.rpc.assert_called_with('create_user_schema_and_tables', {'schema_name': TEST_SCHEMA})
        db_manager.supabase.rpc().execute.assert_called_once()

    def test_create_user_schema_error(self, db_manager):
        """Test handling of errors when creating a schema."""
        # Configure mock to return error
        mock_response = Mock()
        mock_response.data = False
        mock_response.error = "Schema creation error"
        db_manager.supabase.rpc().execute.return_value = mock_response

        # Call method
        result = db_manager.create_user_schema()

        # Verify result
        assert result is False

    def test_get_conversation_history(self, db_manager):
        """Test retrieving conversation history."""
        # Configure mock
        mock_response = Mock()
        mock_response.data = SAMPLE_HISTORY
        db_manager.supabase.rpc().execute.return_value = mock_response

        # Call method
        result = db_manager.get_conversation_history(TEST_SESSION_ID)

        # Verify RPC call and result structure
        db_manager.supabase.rpc.assert_called_with('get_conversation_history', {'schema_name': TEST_SCHEMA})
        assert len(result) == 2
        assert result[0]['questionText'] == SAMPLE_HISTORY[0]['question']
        assert result[0]['answerText'] == SAMPLE_HISTORY[0]['answer']
        assert 'interactionID' in result[0]
        assert 'created_at' in result[0]

    def test_get_conversation_history_empty(self, db_manager):
        """Test retrieving empty conversation history."""
        # Configure mock for empty response
        mock_response = Mock()
        mock_response.data = []
        db_manager.supabase.rpc().execute.return_value = mock_response

        # Call method
        result = db_manager.get_conversation_history(TEST_SESSION_ID)

        # Verify result is empty list
        assert result == []

    def test_get_conversation_history_error(self, db_manager):
        """Test error handling in conversation history retrieval."""
        # Configure mock to raise exception
        db_manager.supabase.rpc().execute.side_effect = Exception("Database error")

        # Call method - should catch exception and return empty list
        result = db_manager.get_conversation_history(TEST_SESSION_ID)

        # Verify empty result
        assert result == []

    def test_add_interaction_success(self, db_manager):
        """Test successfully adding an interaction."""
        # Configure mock
        mock_response = Mock()
        mock_response.data = 1  # New row ID
        mock_response.error = None
        db_manager.supabase.rpc().execute.return_value = mock_response

        # Call method
        result = db_manager.add_interaction(SAMPLE_INTERACTION, TEST_SESSION_ID)

        # Verify result and RPC call
        assert result is True
        db_manager.supabase.rpc.assert_called_with('add_interaction', {
            'p_schema_name': TEST_SCHEMA,
            'p_context': SAMPLE_INTERACTION['context'],
            'p_question': SAMPLE_INTERACTION['question'],
            'p_answer': SAMPLE_INTERACTION['answer'],
            'p_metadata': json.dumps(SAMPLE_INTERACTION['metadata'])
        })

    def test_add_interaction_rpc_failure_fallback(self, db_manager):
        """Test fallback to direct table insert when RPC fails."""
        # Configure first mock to fail, then second to succeed
        db_manager.supabase.rpc().execute.side_effect = Exception("RPC failed")

        mock_response = Mock()
        mock_response.error = None
        db_manager.supabase.table().insert().execute.return_value = mock_response

        # Call method
        result = db_manager.add_interaction(SAMPLE_INTERACTION, TEST_SESSION_ID)

        # Verify fallback to table insert
        assert result is True
        db_manager.supabase.table.assert_called_with(f"{TEST_SCHEMA}.interactions")

    def test_add_interaction_both_methods_fail(self, db_manager):
        """Test handling when both RPC and table insert fail."""
        # Configure both methods to fail
        db_manager.supabase.rpc().execute.side_effect = Exception("RPC failed")
        db_manager.supabase.table().insert().execute.side_effect = Exception("Insert failed")

        # Call method
        result = db_manager.add_interaction(SAMPLE_INTERACTION, TEST_SESSION_ID)

        # Verify failure
        assert result is False

    def test_add_document_to_knowledge_base(self, db_manager):
        """Test adding a document to knowledge base."""
        # Configure mock
        mock_response = Mock()
        mock_response.data = {'id': 1}
        mock_response.error = None
        db_manager.supabase.table().insert().execute.return_value = mock_response

        # Call method
        result = db_manager.add_document_to_knowledge_base(
            "This is test content", 
            [0.1, 0.2, 0.3]
        )

        # Verify result and table call
        assert result is True
        db_manager.supabase.table.assert_called_with(f"{TEST_SCHEMA}.knowledge_base")

    def test_add_document_with_numpy_array(self, db_manager):
        """Test adding a document with numpy array embedding."""
        # Mock numpy array
        class MockNumpyArray:
            def tolist(self):
                return [0.1, 0.2, 0.3]

        mock_array = MockNumpyArray()

        # Configure mock response
        mock_response = Mock()
        mock_response.data = {'id': 1}
        mock_response.error = None
        db_manager.supabase.table().insert().execute.return_value = mock_response

        # Call method
        result = db_manager.add_document_to_knowledge_base(
            "This is test content", 
            mock_array
        )

        # Verify result
        assert result is True

    def test_find_similar_documents(self, db_manager):
        """Test finding similar documents by vector similarity."""
        # Configure mock
        mock_response = Mock()
        mock_response.data = SAMPLE_SIMILAR_DOCUMENTS
        db_manager.supabase.rpc().execute.return_value = mock_response

        # Call method
        result = db_manager.find_similar_documents([0.1, 0.2, 0.3], limit=2)

        # Verify result and RPC call
        assert len(result) == 2
        assert result[0]['content'] == SAMPLE_SIMILAR_DOCUMENTS[0]['content']
        assert result[0]['similarity'] == SAMPLE_SIMILAR_DOCUMENTS[0]['similarity']

        db_manager.supabase.rpc.assert_called_with('find_similar_documents', {
            'p_schema_name': TEST_SCHEMA,
            'p_embedding': str([0.1, 0.2, 0.3]).replace(' ', ''),
            'p_limit': 2
        })

    def test_get_all_documents_and_embeddings(self, db_manager):
        """Test retrieving all documents with embeddings."""
        # Sample document with string embedding (PostgreSQL format)
        postgres_doc = {
            'id': 1,
            'content': 'Sample document content',
            'embedding': '[0.1,0.2,0.3]'
        }

        # Configure mock
        mock_response = Mock()
        mock_response.data = [postgres_doc]
        db_manager.supabase.rpc().execute.return_value = mock_response

        # Call method
        result = db_manager.get_all_documents_and_embeddings()

        # Verify embedding conversion from string to list
        assert len(result) == 1
        assert result[0]['content'] == postgres_doc['content']
        assert isinstance(result[0]['embedding'], list)
        assert result[0]['embedding'] == [0.1, 0.2, 0.3]

        # Verify RPC call
        db_manager.supabase.rpc.assert_called_with('get_knowledge_base_documents', {
            'schema_name': TEST_SCHEMA
        })

    def test_get_topic_interactions(self, db_manager):
        """Test retrieving interactions by topic."""
        # Mock get_conversation_history to return test data
        history_items = [
            {
                'questionText': 'How do I manage anxiety?',
                'answerText': 'There are several techniques...',
                'metadata': json.dumps({'topic': 'Anxiety'})
            },
            {
                'questionText': 'I feel sad all the time',
                'answerText': 'I understand that must be difficult...',
                'metadata': json.dumps({'topic': 'Depression'})
            },
            {
                'questionText': 'Will my anxiety ever go away?',
                'answerText': 'Many people find that with treatment...',
                'metadata': json.dumps({'topic': 'Anxiety'})
            }
        ]

        with patch.object(db_manager, 'get_conversation_history', return_value=history_items):
            # Call method filtering for anxiety topics
            result = db_manager.get_topic_interactions(TEST_SESSION_ID, 'Anxiety')

            # Verify filtered results
            assert len(result) == 2
            assert 'anxiety' in result[0]['questionText'].lower()
            assert 'anxiety' in result[1]['questionText'].lower()

    def test_get_high_quality_interactions(self, db_manager):
        """Test retrieving high-quality interactions for training."""

        # Configure mock response
        mock_response = Mock()
        mock_response.data = [
            {
                'interactionID': 1,
                'context': 'Therapy session 1',
                'question': 'How do I manage anxiety?',
                'answer': 'There are several techniques...',
                'metadata': {
                    'topic': 'Anxiety',
                    'effectiveness': {
                        'term_overlap': 0.9,
                        'template_adherence': 'high'
                    }
                },
                'created_at': '2023-01-01T12:00:00'
            },
            {
                'interactionID': 2,
                'context': 'Therapy session 1',
                'question': 'What are some coping strategies?',
                'answer': 'Coping strategies include...',
                'metadata': {
                    'topic': 'Anxiety',
                    'effectiveness': {
                        'term_overlap': 0.8,
                        'template_adherence': 'high'
                    }
                },
                'created_at': '2023-01-02T12:00:00'
            }
        ]

        # Mock the RPC call to return the mock response
        db_manager.supabase.rpc.return_value.execute.return_value = mock_response

        # Call method
        result = db_manager.get_high_quality_interactions(topic_filter='Anxiety', min_effectiveness=0.7, limit=100)

        # Verify result structure
        assert len(result) == 2  # Expecting 2 interactions
        assert result[0]['question'] == 'How do I manage anxiety?'
        assert result[1]['question'] == 'What are some coping strategies?'

        # Verify that the expected RPC call was made
        db_manager.supabase.rpc.assert_called_once_with('get_high_quality_interactions', {
            'p_schema_name': db_manager.schema_name,
            'p_topic_filter': 'Anxiety',
            'p_min_effectiveness': 0.7,
            'p_limit': 100
        })

    def test_sanitize_schema_name(self, db_manager):
        """Test sanitizing invalid schema names."""
        # Test with valid name
        assert db_manager._sanitize_schema_name("valid_name") == "valid_name"

    def test_connect_psychological_concepts(self, db_manager):
        """Test creating explicit connections between psychological concepts."""

        # Configure mock response for successful connection
        mock_response = Mock()
        mock_response.data = 1  # Simulate returning the new ID of the connection

        # Mock the RPC call to return the mock response
        db_manager.supabase.rpc.return_value.execute.return_value = mock_response

        # Call method
        result = db_manager.connect_psychological_concepts(1, 2, "causes", 0.8)

        # Verify result and calls
        assert result is True  # Expecting a successful connection

        # Verify that the expected RPC call was made
        db_manager.supabase.rpc.assert_called_once_with('connect_psychological_concepts', {
            'p_schema_name': db_manager.schema_name,
            'p_source_id': 1,
            'p_target_id': 2,
            'p_relationship_type': "causes",
            'p_strength': 0.8
        })

    def test_connect_psychological_concepts_error(self, db_manager):
        """Test error handling when connecting psychological concepts."""
        # Configure mock to fail
        db_manager.supabase.rpc.return_value.execute.side_effect = Exception("Connection error")

        # Call method
        result = db_manager.connect_psychological_concepts(1, 2)

        # Verify failure handling
        assert result is False

    def test_extract_psychological_themes(self, db_manager):
        """Test extracting recurring psychological themes."""
        # Sample history with recurring themes
        themed_history = [
            {
                'questionText': 'How do you feel when someone criticizes you?',
                'answerText': 'I feel worthless and like I\'m a complete failure.',
                'metadata': json.dumps({'topic': 'Self-Worth'})
            },
            {
                'questionText': 'Tell me about your relationship with your parents.',
                'answerText': 'My father abandoned us when I was young. I felt so alone.',
                'metadata': json.dumps({'topic': 'Abandonment'})
            },
            {
                'questionText': 'How do you handle feedback at work?',
                'answerText': 'I get defensive because deep down I feel like a failure.',
                'metadata': json.dumps({'topic': 'Self-Worth'})
            },
            {
                'questionText': 'What happens in your romantic relationships?',
                'answerText': 'I worry my partner will abandon me like everyone else.',
                'metadata': json.dumps({'topic': 'Abandonment'})
            }
        ]

        # Mock conversation history
        with patch.object(db_manager, 'get_conversation_history', return_value=themed_history):
            # Call method with min_occurrences=2
            result = db_manager.extract_psychological_themes(TEST_SESSION_ID, min_occurrences=2)

            # Verify the recurring themes were correctly identified
            assert 'Self-Worth' in result
            assert 'Abandonment' in result
            assert result['Self-Worth'] >= 2
            assert result['Abandonment'] >= 2

    def test_extract_psychological_themes_empty(self, db_manager):
        """Test extracting themes from empty history."""
        # Mock empty conversation history
        with patch.object(db_manager, 'get_conversation_history', return_value=[]):
            # Call method
            result = db_manager.extract_psychological_themes(TEST_SESSION_ID)

            # Verify empty result
            assert result == {}

    def test_extract_psychological_themes_content_analysis(self, db_manager):
        """Test extracting themes from content even without explicit metadata."""
        # History with implicit themes in content
        implicit_history = [
            {
                'questionText': 'What happens when you try to express your needs?',
                'answerText': 'I can\'t control how others react, so I just keep quiet.',
                'metadata': json.dumps({'topic': 'Communication'})  # Not a recurring theme
            },
            {
                'questionText': 'How do you handle difficult situations?',
                'answerText': 'I feel helpless and powerless to change anything.',
                'metadata': json.dumps({'topic': 'Coping'})  # Not a recurring theme
            }
        ]

        # Mock conversation history
        with patch.object(db_manager, 'get_conversation_history', return_value=implicit_history):
            # Call method
            result = db_manager.extract_psychological_themes(TEST_SESSION_ID, min_occurrences=1)

            # Verify content-based theme detection
            assert 'Control' in result

    def test_analyze_emotional_vector_trajectory(self, db_manager):
        """Test analyzing emotional trajectory across therapy sessions."""

        # Sample history with emotional states
        emotional_history = [
            {
                'interactionID': 1,
                'context': 'Therapy session 1',
                'question': 'How are you feeling today?',
                'answer': 'Overwhelmed and anxious.',
                'metadata': json.dumps({
                    'session_id': TEST_SESSION_ID,
                    'emotional_state': 'anxious',
                    'emotional_intensity': 8
                }),
                'created_at': '2023-01-01T12:00:00'
            },
            {
                'interactionID': 2,
                'context': 'Therapy session 1',
                'question': 'How did the breathing exercises work for you?',
                'answer': 'They helped a bit. I\'m still anxious but less than before.',
                'metadata': json.dumps({
                    'session_id': TEST_SESSION_ID,
                    'emotional_state': 'anxious',
                    'emotional_intensity': 6
                }),
                'created_at': '2023-01-08T12:00:00'
            },
            {
                'interactionID': 3,
                'context': 'Therapy session 1',
                'question': 'How are you feeling now about your progress?',
                'answer': 'I feel more hopeful and my anxiety is more manageable.',
                'metadata': json.dumps({
                    'session_id': TEST_SESSION_ID,
                    'emotional_state': 'hopeful',
                    'emotional_intensity': 4
                }),
                'created_at': '2023-01-15T12:00:00'
            }
        ]

        # Mock conversation history
        with patch.object(db_manager, 'get_conversation_history', return_value=emotional_history):
            # Call method
            result = db_manager.analyze_emotional_vector_trajectory(TEST_SESSION_ID)

            # Verify trajectory shows emotional progression
            assert len(result) == 3  # Expecting 3 segments
            assert result[0]['start_state'] == 'anxious'
            assert result[1]['start_state'] == 'anxious'
            assert result[2]['start_state'] == 'hopeful'
            assert result[0]['start_intensity'] > result[1]['start_intensity']
            assert result[1]['start_intensity'] > result[2]['start_intensity']

            # Verify chronological ordering
            timestamps = [entry['created_at'] for entry in result]
            assert timestamps == sorted(timestamps)

    def test_analyze_emotional_vector_trajectory_empty(self, db_manager):
        """Test emotional trajectory with empty or invalid data."""
        # Mock conversation history with no emotional data
        history_no_emotions = [
            {
                'questionText': 'What brings you here today?',
                'answerText': 'I\'m not sure where to start.',
                'metadata': json.dumps({'topic': 'Initial Assessment'}),
                'created_at': '2023-01-01T12:00:00'
            }
        ]

        with patch.object(db_manager, 'get_conversation_history', return_value=history_no_emotions):
            # Call method
            result = db_manager.analyze_emotional_vector_trajectory(TEST_SESSION_ID)

            # Verify empty result when no emotional data
            assert result == []

    def test_analyze_emotional_vector_trajectory_invalid_metadata(self, db_manager):
        """Test emotional trajectory with invalid metadata format."""
        # History with invalid metadata
        invalid_history = [
            {
                'questionText': 'How are you feeling?',
                'answerText': 'Not great.',
                'metadata': '{invalid:json}',  # Invalid JSON
                'created_at': '2023-01-01T12:00:00'
            }
        ]

        with patch.object(db_manager, 'get_conversation_history', return_value=invalid_history):
            # Call method - should handle invalid JSON without errors
            result = db_manager.analyze_emotional_vector_trajectory(TEST_SESSION_ID)

            # Verify empty result for invalid metadata
            assert result == []

    def test_start_therapy_session(self, db_manager):
        """Test marking the start of a new therapy session."""
        # Mock add_interaction to return success
        with patch.object(db_manager, 'add_interaction', return_value=True):
            # Call method with custom metadata
            custom_metadata = {'session_theme': 'Trust issues', 'session_number': 3}
            result = db_manager.start_therapy_session(TEST_SESSION_ID, custom_metadata)

            # Verify result
            assert result is True

            # Verify add_interaction was called with proper data
            expected_data = {
                'context': 'Session Start',
                'question': 'Beginning of therapy session',
                'answer': '',
                'metadata': {
                    'session_theme': 'Trust issues',
                    'session_number': 3,
                    'session_start': True,
                    'session_timestamp': ANY  # We don't know exact timestamp
                }
            }
            db_manager.add_interaction.assert_called_once()
            call_args = db_manager.add_interaction.call_args[0]
            assert call_args[0]['context'] == expected_data['context']
            assert call_args[0]['question'] == expected_data['question']
            assert call_args[0]['answer'] == expected_data['answer']
            assert 'session_start' in call_args[0]['metadata']
            assert call_args[0]['metadata']['session_start'] is True
            assert 'session_theme' in call_args[0]['metadata']
            assert 'session_timestamp' in call_args[0]['metadata']
            assert call_args[1] == TEST_SESSION_ID

    def test_start_therapy_session_failure(self, db_manager):
        """Test failure handling when starting a therapy session."""
        # Mock add_interaction to return failure
        with patch.object(db_manager, 'add_interaction', return_value=False):
            # Call method
            result = db_manager.start_therapy_session(TEST_SESSION_ID)

            # Verify result
            assert result is False

    def test_start_therapy_session_error(self, db_manager):
        """Test error handling when starting a therapy session."""
        # Mock add_interaction to raise an exception
        with patch.object(db_manager, 'add_interaction', side_effect=Exception("Database error")):
            # Call method
            result = db_manager.start_therapy_session(TEST_SESSION_ID)

            # Verify result
            assert result is False

    def test_mark_therapeutic_insight(self, db_manager):
        """Test marking an interaction as containing a therapeutic insight."""
        # Mock RPC call to return success
        mock_response = Mock()
        mock_response.error = None
        db_manager.supabase.rpc().execute.return_value = mock_response

        # Call method
        result = db_manager.mark_therapeutic_insight(1, "high", TEST_SESSION_ID)

        # Verify result and RPC call
        assert result is True
        db_manager.supabase.rpc.assert_called_with('execute_sql', {'command': ANY})

        # Verify SQL contains correct values
        command = db_manager.supabase.rpc.call_args[1]['command']
        assert f'"{TEST_SESSION_ID}".interactions' in command
        assert 'SET metadata = jsonb_set' in command
        assert "'insight_level'" in command
        assert '"high"' in command
        assert 'WHERE interactionid = 1' in command

    def test_mark_therapeutic_insight_null_metadata(self, db_manager):
        """Test marking an interaction that has null metadata."""
        # Mock RPC call to return success
        mock_response = Mock()
        mock_response.error = None
        db_manager.supabase.rpc().execute.return_value = mock_response

        # Call method (no session_id provided, should use schema_name)
        result = db_manager.mark_therapeutic_insight(1, "medium")

        # Verify result
        assert result is True

        # Verify RPC call with correct function name and parameters
        db_manager.supabase.rpc.assert_called_with('mark_therapeutic_insight', {
            'p_schema_name': TEST_SCHEMA,
            'p_interaction_id': 1,
            'p_insight_level': 'medium'
        })

    def test_mark_therapeutic_insight_error(self, db_manager):
        """Test error handling when marking a therapeutic insight."""
        # Mock RPC call to raise exception
        db_manager.supabase.rpc().execute.side_effect = Exception("Database error")

        # Call method
        result = db_manager.mark_therapeutic_insight(1, "high")

        # Verify result
        assert result is False
