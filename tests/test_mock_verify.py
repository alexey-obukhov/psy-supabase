def test_mock_db_manager_with_spy(mock_db_manager_with_spy):
    """Test that our mock_db_manager_with_spy fixture works correctly."""
    # The fixture is automatically provided by pytest, no need to import it

    # Call save_interaction on the mock
    mock_db_manager_with_spy.save_interaction(
        context="test_context",
        question="test_question",
        answer="test_answer",
        metadata={"test": True},
        session_id="test_session",
    )

    # Verify it was called
    mock_db_manager_with_spy.save_interaction.assert_called_once()

    # Get the context using helper method
    context = mock_db_manager_with_spy.get_last_context()
    assert context == "test_context"

    # Print debug info
    print(f"Mock DB: {mock_db_manager_with_spy}")
    print(f"save_interaction mock: {mock_db_manager_with_spy.save_interaction}")
    print(f"save_interaction call count: {mock_db_manager_with_spy.save_interaction.call_count}")
    print(f"save_interaction call args: {mock_db_manager_with_spy.save_interaction.call_args}")
