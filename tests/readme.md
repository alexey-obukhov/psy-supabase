# Psy Supabase Tests

This directory contains comprehensive tests for the Psy Supabase psychological RAG system. The tests ensure both technical correctness and therapeutic effectiveness of the system.

## Test Organization

- `test_rag_processor.py` - Tests for the core RAG processor with therapeutic capabilities
- `test_database.py` - Tests for vector database operations and conversation storage
- `conftest.py` - Shared test fixtures and constants

## Testing Philosophy

Our testing approach combines technical validation with therapeutic efficacy verification:

1. **Technical Tests**: Ensure embedding generation, vector similarity, and database operations work correctly
2. **Therapeutic Tests**: Validate psychological reasoning, approach selection, and therapeutic effectiveness
3. **Integration Tests**: Verify all components work together to deliver appropriate responses

## Running Tests

```bash
# Run all tests
pytest

# Run only RAGProcessor tests
pytest tests/test_rag_processor.py

# Run specific test categories
pytest tests/test_rag_processor.py::TestRAGProcessor::test_detect_pain_points_from_embedding

# Run with increased verbosity
pytest -v

# Generate test coverage report
pytest --cov=psy_supabase
```

## Key Test Categories

### Pain Point Detection Tests

These tests validate the system's ability to identify psychological concerns and select appropriate therapeutic approaches:

- Detection of fixation patterns in user questions
- Recognition of recurring psychological themes
- Appropriate therapeutic approach selection based on emotional states
- Handling of ambiguous psychological concerns

### Vector Retrieval Tests

Tests for the vector similarity and knowledge retrieval systems:

- Accurate document similarity matching
- Vector caching and retrieval optimization
- Threshold filtering for relevant content
- Edge cases with low-similarity or irrelevant documents

### Response Generation Tests

Tests that validate therapeutic response quality:

- Template selection logic for different psychological scenarios
- Context incorporation from knowledge base
- Conversation history integration
- Appropriate emotional tone in responses

### Safety Feature Tests

Tests for system safety and reliability:

- Toxic content detection and appropriate responses
- Crisis detection and handling
- Error recovery mechanisms
- Performance under constrained resources

## Test Implementation Details

### Fixtures

Our tests use comprehensive fixtures defined in `conftest.py`:

- `TEST_USER_ID`, `TEST_SCHEMA`, `TEST_SESSION_ID` - Standard IDs for testing
- `TEST_URL`, `TEST_KEY` - API endpoint constants
- `COMPLEX_METADATA` - Standard metadata structure for testing

### Mock Strategy

Tests use strategic mocking to isolate components:

```python
# Example of database manager mocking
@pytest.fixture
def mock_db_manager():
    mock = Mock(spec=DatabaseManager)
    mock.schema_name = TEST_SCHEMA
    return mock

# Example of text generator mocking
@pytest.fixture
def mock_text_generator():
    mock = Mock(spec=TextGenerator)
    mock.is_toxic.return_value = False
    mock.generate_therapeutic_response_with_dynamic_retrieval.return_value = "This is a therapeutic response"
    return mock
```

## Extending the Tests

When adding new features to the system, corresponding tests should be added:

1. Create unit tests for new components
2. Add integration tests for interaction with existing components 
3. Include edge cases and failure scenarios
4. For therapeutic features, test both technical operation and psychological soundness

### Test Template

```python
def test_new_feature(self, rag_processor):
    """Test description that explains purpose and psychological significance."""
    # Setup test data
    input_data = "Test input"
    
    # Execute test
    result = rag_processor.new_feature(input_data)
    
    # Verify technical correctness
    assert result["success"] == True
    
    # Verify therapeutic appropriateness
    assert "supportive_element" in result["response"]
```

## Code Coverage Goals

By maintaining comprehensive test coverage, we ensure the system provides reliable, psychologically-informed support while maintaining technical excellence.
