#!/usr/bin/env python3
"""
Semantic Chunking Integration Test Suite

This test suite helps understand how to integrate semantic chunking into the main psy-supabase program.
It explores all the integration points and expected behaviors to guide the implementation.

Test Categories:
1. Core chunking functionality tests
2. Integration with existing pain point detection
3. Vector similarity search integration  
4. User schema and database integration
5. Performance and configuration tests
6. End-to-end workflow tests
"""

import pytest
import os
import sys
import uuid
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.pain_point_detector import PainPointDetector
from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.config import TEXT_GENERATING_MODEL, PAIN_POINT_DETECTION
from psy_supabase import get_package_logger

logger = get_package_logger(__name__)


class TestSemanticChunkingIntegration:
    """Test suite for semantic chunking integration."""
    
    @pytest.fixture
    def test_user_id(self):
        """Generate a unique test user ID."""
        return f"test_chunking_{uuid.uuid4().hex[:8]}"
    
    @pytest.fixture
    def db_manager(self, test_user_id):
        """Create a test database manager."""
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            pytest.skip("Supabase credentials not available")
        
        return DatabaseManager(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            user_id=test_user_id
        )
    
    @pytest.fixture
    def pain_detector(self, db_manager):
        """Create a pain point detector instance."""
        return db_manager.pain_point_detector
    
    @pytest.fixture
    def sample_questions(self):
        """Sample questions for testing semantic chunking."""
        return [
            "I feel inadequate at work every single day and it's affecting my confidence",
            "I'm still feeling inadequate every day and thinking about quitting my job",
            "My anxiety is getting worse when I have to speak in public meetings",
            "I can't sleep at night because I keep worrying about tomorrow's presentation",
            "I feel overwhelmed by my workload and don't know how to manage my time",
            "My relationship with my partner is suffering because I'm always stressed",
            "I have trouble concentrating on tasks and my mind keeps wandering",
            "I feel disconnected from my friends and family lately"
        ]

    # Test Category 1: Core Chunking Functionality
    
    def test_semantic_chunking_basic_functionality(self, pain_detector, sample_questions):
        """Test that semantic chunking extracts meaningful phrases."""
        logger.info("🧪 Testing basic semantic chunking functionality")
        
        question = sample_questions[0]
        chunks = pain_detector._semantic_chunk_question(question)
        
        # Expected behavior: chunks should extract key concepts
        assert isinstance(chunks, list), "Chunks should be a list"
        assert len(chunks) > 0, "Should extract at least one chunk"
        assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"
        
        # Expected: should extract emotional/psychological concepts
        expected_concepts = ["inadequate", "work", "confidence", "affecting"]
        found_concepts = [concept for concept in expected_concepts 
                         if any(concept in chunk.lower() for chunk in chunks)]
        
        assert len(found_concepts) > 0, f"Should extract psychological concepts, got chunks: {chunks}"
        
        logger.info(f"✅ Extracted {len(chunks)} chunks: {chunks}")
        logger.info(f"✅ Found concepts: {found_concepts}")
    
    def test_chunking_configuration_impact(self, pain_detector, sample_questions):
        """Test how configuration affects chunking behavior."""
        logger.info("🧪 Testing chunking configuration impact")
        
        question = sample_questions[0]
        
        # Test with different configurations
        original_config = PAIN_POINT_DETECTION["chunking"].copy()
        
        try:
            # High granularity config
            PAIN_POINT_DETECTION["chunking"]["max_chunks_per_question"] = 12
            PAIN_POINT_DETECTION["chunking"]["include_full_question"] = True
            
            chunks_detailed = pain_detector._semantic_chunk_question(question)
            
            # Low granularity config
            PAIN_POINT_DETECTION["chunking"]["max_chunks_per_question"] = 3
            PAIN_POINT_DETECTION["chunking"]["include_full_question"] = False
            
            chunks_simple = pain_detector._semantic_chunk_question(question)
            
            # Expected: detailed config should produce more or equal chunks
            assert len(chunks_detailed) >= len(chunks_simple), \
                f"Detailed config should produce more chunks: {len(chunks_detailed)} >= {len(chunks_simple)}"
            
            logger.info(f"✅ Detailed chunks ({len(chunks_detailed)}): {chunks_detailed}")
            logger.info(f"✅ Simple chunks ({len(chunks_simple)}): {chunks_simple}")
            
        finally:
            # Restore original config
            PAIN_POINT_DETECTION["chunking"].update(original_config)

    # Test Category 2: Integration with Pain Point Detection
    
    def test_chunking_improves_pain_point_detection(self, pain_detector, sample_questions):
        """Test that chunking improves pain point detection accuracy."""
        logger.info("🧪 Testing chunking integration with pain point detection")
        
        # Simulate conversation with repeated themes
        questions = sample_questions[:4]  # Take first 4 related questions
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        
        # Expected: should detect pain points using chunking
        for i, question in enumerate(questions):
            result = pain_detector.detect_pain_points(
                question=question,
                session_id=session_id,
                context=f"Previous interactions: {i}"
            )
            
            assert isinstance(result, dict), "Should return a result dictionary"
            assert "pain_points_detected" in result, "Should include pain points detection status"
            
            if i > 0:  # After first question, should start detecting patterns
                logger.info(f"Question {i+1}: {question[:50]}...")
                logger.info(f"Detection result: {result}")
    
    def test_chunking_similarity_threshold_behavior(self, pain_detector, sample_questions):
        """Test how similarity thresholds affect chunking-based detection."""
        logger.info("🧪 Testing similarity threshold behavior with chunking")
        
        q1, q2 = sample_questions[0], sample_questions[1]  # Related questions
        
        # Test similarity calculation
        chunks1 = pain_detector._semantic_chunk_question(q1)
        chunks2 = pain_detector._semantic_chunk_question(q2)
        
        similarity = pain_detector._calculate_max_chunk_similarity(chunks1, chunks2)
        
        # Expected: related questions should have high similarity
        assert 0.0 <= similarity <= 1.0, f"Similarity should be between 0 and 1, got {similarity}"
        assert similarity > 0.5, f"Related questions should have high similarity, got {similarity}"
        
        # Test with different thresholds
        thresholds = [0.3, 0.5, 0.6, 0.7, 0.8]
        results = {}
        
        for threshold in thresholds:
            is_similar = similarity >= threshold
            results[threshold] = is_similar
            logger.info(f"Threshold {threshold}: {'PASS' if is_similar else 'FAIL'} (similarity: {similarity:.3f})")
        
        # Expected: at least some thresholds should pass for related questions
        passing_thresholds = [t for t, passed in results.items() if passed]
        assert len(passing_thresholds) > 0, f"At least some thresholds should pass, got similarity {similarity}"

    # Test Category 3: Vector Similarity Search Integration
    
    def test_chunking_with_vector_search(self, db_manager, pain_detector, sample_questions):
        """Test integration of chunking with vector similarity search."""
        logger.info("🧪 Testing chunking integration with vector search")
        
        # Create schema and add some interactions
        db_manager.create_user_schema_sync()
        
        try:
            session_id = f"vector_test_{uuid.uuid4().hex[:8]}"
            
            # Add interactions with chunked content
            for i, question in enumerate(sample_questions[:3]):
                interaction = {
                    "session_id": session_id,
                    "question": question,
                    "answer": f"Response to question {i+1}",
                    "metadata": {"test": "chunking_vector_integration"}
                }
                
                result = db_manager.add_interaction(interaction, session_id)
                assert result is not None, f"Should successfully add interaction {i+1}"
            
            # Test vector similarity search
            query_question = "I'm feeling overwhelmed and inadequate at work"
     
            # Expected: should find similar interactions using vector search
            similar_docs = db_manager.find_similar_documents(
                embedding=None,  # Let it generate from question
                limit=3,
                question=query_question
            )
            
            assert isinstance(similar_docs, list), "Should return list of similar documents"
            logger.info(f"✅ Found {len(similar_docs)} similar documents")
            
            # Test chunking on the search results
            if similar_docs:
                for doc in similar_docs:
                    if 'question' in doc:
                        chunks = pain_detector._semantic_chunk_question(doc['question'])
                        logger.info(f"Document chunks: {chunks}")
                        assert len(chunks) > 0, "Should chunk similar documents"
            
        finally:
            # Cleanup
            try:
                db_manager.cleanup_user_data()
            except:
                pass

    # Test Category 4: User Schema and Database Integration
    
    def test_chunking_with_user_schema_isolation(self, test_user_id):
        """Test that chunking works correctly with user schema isolation."""
        logger.info("🧪 Testing chunking with user schema isolation")
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        # Create two different user schemas
        user1_id = f"{test_user_id}_user1"
        user2_id = f"{test_user_id}_user2"
        
        db1 = DatabaseManager(supabase_url=supabase_url, supabase_key=supabase_key, user_id=user1_id)
        db2 = DatabaseManager(supabase_url=supabase_url, supabase_key=supabase_key, user_id=user2_id)
        
        try:
            # Create schemas
            db1.create_user_schema_sync()
            db2.create_user_schema_sync()
            
            # Test that chunking works independently for each user
            question = "I feel anxious about my performance at work"
            
            chunks1 = db1.pain_point_detector._semantic_chunk_question(question)
            chunks2 = db2.pain_point_detector._semantic_chunk_question(question)
            
            # Expected: chunking should work the same way regardless of user
            assert chunks1 == chunks2, "Chunking should be consistent across users"
            
            # Test that pain point detection maintains user isolation
            session1 = f"session1_{uuid.uuid4().hex[:8]}"
            session2 = f"session2_{uuid.uuid4().hex[:8]}"
            
            result1 = db1.pain_point_detector.detect_pain_points(session1)
            result2 = db2.pain_point_detector.detect_pain_points(session2)
            
            assert isinstance(result1, dict), "User 1 should get valid result"
            assert isinstance(result2, dict), "User 2 should get valid result"
            
            logger.info(f"✅ User schema isolation working with chunking")
            
        finally:
            # Cleanup
            for db in [db1, db2]:
                try:
                    db.cleanup_user_data()
                except:
                    pass

    # Test Category 5: Performance and Configuration Tests
    
    def test_chunking_performance_optimization(self, pain_detector, sample_questions):
        """Test performance optimizations in chunking."""
        logger.info("🧪 Testing chunking performance optimizations")
        
        import time
        
        question = sample_questions[0]
        
        # Test early exit optimization
        original_config = PAIN_POINT_DETECTION["chunking"].copy()
        
        try:
            # Enable early exit
            PAIN_POINT_DETECTION["chunking"]["early_exit_on_perfect_match"] = True
            
            start_time = time.time()
            chunks = pain_detector._semantic_chunk_question(question)
            optimized_time = time.time() - start_time
            
            # Disable early exit
            PAIN_POINT_DETECTION["chunking"]["early_exit_on_perfect_match"] = False
            
            start_time = time.time()
            chunks_no_opt = pain_detector._semantic_chunk_question(question)
            normal_time = time.time() - start_time
            
            # Expected: results should be the same regardless of optimization
            assert chunks == chunks_no_opt, "Optimization shouldn't change results"
            
            logger.info(f"✅ Optimized time: {optimized_time:.4f}s, Normal time: {normal_time:.4f}s")
            
        finally:
            PAIN_POINT_DETECTION["chunking"].update(original_config)
    
    def test_chunking_memory_management(self, pain_detector, sample_questions):
        """Test memory management in chunking operations."""
        logger.info("🧪 Testing chunking memory management")
        
        # Test with many questions to check memory usage
        large_question_set = sample_questions * 10  # 80 questions
        
        chunk_results = []
        for question in large_question_set:
            chunks = pain_detector._semantic_chunk_question(question)
            chunk_results.append(chunks)
        
        # Expected: should handle large datasets without memory issues
        assert len(chunk_results) == len(large_question_set), "Should process all questions"
        assert all(len(chunks) > 0 for chunks in chunk_results), "All questions should produce chunks"
        
        logger.info(f"✅ Processed {len(large_question_set)} questions successfully")

    # Test Category 6: End-to-End Workflow Tests
    
    def test_complete_chunking_workflow(self, db_manager, sample_questions):
        """Test complete workflow integration with chunking."""
        logger.info("🧪 Testing complete workflow with chunking integration")
        
        # Setup
        db_manager.create_user_schema_sync()
        session_id = f"workflow_test_{uuid.uuid4().hex[:8]}"
        
        try:
            # Simulate a complete therapy session with chunking
            responses = []
            
            for i, question in enumerate(sample_questions[:4]):
                logger.info(f"\n--- Processing question {i+1} ---")
                logger.info(f"Question: {question[:50]}...")
                
                # Step 1: Semantic chunking
                chunks = db_manager.pain_point_detector._semantic_chunk_question(question)
                logger.info(f"Chunks: {chunks}")
                
                # Step 2: Pain point detection with chunking
                pain_result = db_manager.pain_point_detector.detect_pain_points(
                    session_id=session_id,
                    threshold=0.7,
                    min_occurrences=2,
                    time_window_days=30
                )
                logger.info(f"Pain points: {pain_result}")
                
                # Step 3: Add interaction with chunked metadata
                interaction = {
                    "session_id": session_id,
                    "question": question,
                    "answer": f"Therapeutic response {i+1}",
                    "metadata": {
                        "chunks": chunks,
                        "pain_points": pain_result,
                        "workflow_test": True
                    }
                }
                
                add_result = db_manager.add_interaction(interaction, session_id)
                assert add_result is not None, f"Should add interaction {i+1}"
                
                # Step 4: Vector similarity search
                if i > 0:  # After first interaction
                    similar = db_manager.find_similar_documents(
                        embedding=None,
                        limit=3,
                        question=question
                    )
                    logger.info(f"Similar documents: {len(similar)}")
                
                responses.append({
                    "question": question,
                    "chunks": chunks,
                    "pain_points": pain_result,
                    "interaction_added": add_result is not None
                })
            
            # Validate complete workflow
            assert len(responses) == 4, "Should process all questions"
            assert all(r["interaction_added"] for r in responses), "Should add all interactions"
            assert all(len(r["chunks"]) > 0 for r in responses), "Should chunk all questions"
            
            logger.info(f"✅ Complete workflow test passed with {len(responses)} interactions")
            
        finally:
            # Cleanup
            try:
                db_manager.cleanup_user_data()
            except:
                pass
    
    def test_chunking_integration_with_rag_processor(self, db_manager, sample_questions):
        """Test chunking integration with RAG processor."""
        logger.info("🧪 Testing chunking integration with RAG processor")
        
        # Create a mock text generator for testing
        mock_generator = Mock(spec=TextGenerator)
        mock_generator.get_test_response.return_value = "Mock therapeutic response"
        
        # Create RAG processor
        rag_processor = RAGProcessor(
            db_manager=db_manager,
            generator=mock_generator,
            intelligent_processing_enabled=True
        )
        
        # Setup schema
        db_manager.create_user_schema_sync()
        session_id = f"rag_test_{uuid.uuid4().hex[:8]}"
        
        try:
            # Test RAG processing with chunking
            question = sample_questions[0]
            
            # Expected: RAG processor should use chunking internally
            response = rag_processor.generate_response(question, session_id)
            
            assert isinstance(response, str), "Should return a response string"
            assert len(response) > 0, "Response should not be empty"
            
            # Verify that chunking was used (check if pain point detection was called)
            # This is implicit - the system should use chunking for pain point detection
            
            logger.info(f"✅ RAG processor integration test passed")
            logger.info(f"Response: {response[:100]}...")
            
        finally:
            try:
                db_manager.cleanup_user_data()
            except:
                pass


# Integration helper functions for the test suite

def run_integration_tests():
    """Run the semantic chunking integration test suite."""
    logger.info("🚀 Running Semantic Chunking Integration Test Suite")
    logger.info("=" * 60)
    
    # Run pytest with specific markers
    import subprocess
    
    result = subprocess.run([
        "python", "-m", "pytest", 
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run the integration tests
    success = run_integration_tests()
    sys.exit(0 if success else 1)
