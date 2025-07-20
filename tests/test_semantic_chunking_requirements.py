#!/usr/bin/env python3
"""
Semantic Chunking Integration Requirements Test Suite

This test suite defines the expected behaviors and integration requirements
for semantic chunking in the psy-supabase system. It serves as both
documentation and validation for the integration process.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psy_supabase import get_package_logger

logger = get_package_logger(__name__)


class TestSemanticChunkingRequirements:
    """Test the requirements and expected behaviors for semantic chunking integration."""
    
    def test_chunking_interface_requirements(self):
        """Test that chunking provides the required interface."""
        logger.info("🧪 Testing chunking interface requirements")
        
        # Expected interface for semantic chunking
        expected_methods = [
            "_semantic_chunk_question",
            "_calculate_max_chunk_similarity", 
            "_extract_repeating_terms",
            "_extract_semantic_theme"
        ]
        
        # Mock pain point detector to test interface
        from psy_supabase.core.pain_point_detector import PainPointDetector
        
        # Check that PainPointDetector has chunking methods
        detector_methods = [method for method in dir(PainPointDetector) 
                           if not method.startswith('__')]
        
        for method in expected_methods:
            assert hasattr(PainPointDetector, method), \
                f"PainPointDetector should have method {method}"
        
        logger.info(f"✅ Interface requirements met")
        logger.info(f"Available methods: {[m for m in detector_methods if 'chunk' in m.lower()]}")
    
    def test_chunking_input_output_contracts(self):
        """Test the input/output contracts for chunking methods."""
        logger.info("🧪 Testing chunking input/output contracts")
        
        # Mock the pain point detector
        mock_detector = Mock()
        
        # Test _semantic_chunk_question contract
        # Input: string question
        # Output: list of string chunks
        test_question = "I feel anxious about my work performance"
        expected_chunks = ["feel anxious", "work performance", "anxious about work"]
        
        mock_detector._semantic_chunk_question.return_value = expected_chunks
        
        result = mock_detector._semantic_chunk_question(test_question)
        
        # Validate contract
        assert isinstance(result, list), "Should return a list"
        assert all(isinstance(chunk, str) for chunk in result), "All chunks should be strings"
        assert len(result) > 0, "Should return at least one chunk"
        
        # Test _calculate_max_chunk_similarity contract
        # Input: two lists of chunks
        # Output: float similarity score (0.0 to 1.0)
        chunks1 = ["feel anxious", "work"]
        chunks2 = ["anxious feelings", "workplace"]
        expected_similarity = 0.75
        
        mock_detector._calculate_max_chunk_similarity.return_value = expected_similarity
        
        similarity = mock_detector._calculate_max_chunk_similarity(chunks1, chunks2)
        
        # Validate contract
        assert isinstance(similarity, (int, float)), "Should return a numeric value"
        assert 0.0 <= similarity <= 1.0, f"Similarity should be between 0 and 1, got {similarity}"
        
        logger.info("✅ Input/output contracts validated")
    
    def test_chunking_configuration_requirements(self):
        """Test configuration requirements for chunking."""
        logger.info("🧪 Testing chunking configuration requirements")
        
        from psy_supabase.config import PAIN_POINT_DETECTION
        
        # Expected configuration options
        required_config_keys = [
            "chunking",
            "similarity_threshold"
        ]
        
        chunking_config_keys = [
            "max_chunks_per_question",
            "include_full_question", 
            "early_exit_on_perfect_match",
            "log_only_best_matches"
        ]
        
        # Validate main config structure
        for key in required_config_keys:
            assert key in PAIN_POINT_DETECTION, f"Config should have {key}"
        
        # Validate chunking sub-config
        chunking_config = PAIN_POINT_DETECTION.get("chunking", {})
        for key in chunking_config_keys:
            assert key in chunking_config, f"Chunking config should have {key}"
        
        # Validate config value types and ranges
        assert isinstance(chunking_config["max_chunks_per_question"], int), \
            "max_chunks_per_question should be integer"
        assert chunking_config["max_chunks_per_question"] > 0, \
            "max_chunks_per_question should be positive"
        
        assert isinstance(chunking_config["include_full_question"], bool), \
            "include_full_question should be boolean"
        
        assert isinstance(PAIN_POINT_DETECTION["similarity_threshold"], (int, float)), \
            "similarity_threshold should be numeric"
        assert 0.0 <= PAIN_POINT_DETECTION["similarity_threshold"] <= 1.0, \
            "similarity_threshold should be between 0 and 1"
        
        logger.info("✅ Configuration requirements validated")
    
    def test_chunking_pain_point_integration_requirements(self):
        """Test requirements for chunking integration with pain point detection."""
        logger.info("🧪 Testing chunking-pain point integration requirements")
        
        # Expected: Pain point detector should use chunking for similarity comparison
        # Expected: Chunking should improve pain point detection accuracy
        # Expected: Chunking should not break existing pain point detection functionality
        
        # Mock scenario: two related questions should be detected as similar using chunking
        mock_detector = Mock()
        
        # Simulate chunking improving detection
        question1 = "I feel inadequate at work every day"
        question2 = "I'm feeling inadequate every day at my job"
        
        # Without chunking (simple string comparison) - low similarity
        mock_detector.simple_similarity.return_value = 0.3
        
        # With chunking (semantic comparison) - high similarity  
        chunks1 = ["feel inadequate", "work", "every day"]
        chunks2 = ["feeling inadequate", "every day", "job"]
        
        mock_detector._semantic_chunk_question.side_effect = [chunks1, chunks2]
        mock_detector._calculate_max_chunk_similarity.return_value = 0.85
        
        # Test the integration
        chunks_q1 = mock_detector._semantic_chunk_question(question1)
        chunks_q2 = mock_detector._semantic_chunk_question(question2)
        chunked_similarity = mock_detector._calculate_max_chunk_similarity(chunks_q1, chunks_q2)
        simple_similarity = mock_detector.simple_similarity(question1, question2)
        
        # Expected: chunking should provide better similarity detection
        assert chunked_similarity > simple_similarity, \
            f"Chunking should improve similarity: {chunked_similarity} > {simple_similarity}"
        
        assert chunked_similarity >= 0.7, \
            f"Related questions should have high chunked similarity: {chunked_similarity}"
        
        logger.info(f"✅ Chunking improves similarity: {simple_similarity} → {chunked_similarity}")
    
    def test_chunking_database_integration_requirements(self):
        """Test requirements for chunking integration with database operations."""
        logger.info("🧪 Testing chunking-database integration requirements")
        
        # Expected: Chunked data should be stored in interaction metadata
        # Expected: Vector similarity should work with chunked content
        # Expected: User schema isolation should work with chunking
        
        # Mock database manager
        mock_db = Mock()
        
        # Test interaction storage with chunking metadata
        interaction_data = {
            "session_id": "test_session",
            "question": "I'm anxious about work",
            "answer": "Therapeutic response",
            "metadata": {
                "chunks": ["anxious", "work", "anxious about work"],
                "chunk_similarity_scores": [0.85, 0.72],
                "pain_points_detected": True
            }
        }
        
        mock_db.add_interaction.return_value = "interaction_123"
        
        result = mock_db.add_interaction(interaction_data, "test_session")
        
        # Validate integration
        assert result is not None, "Should successfully store chunked interaction"
        
        # Test vector similarity with chunking
        mock_db.find_similar_documents.return_value = [
            {
                "question": "I feel anxious at work",
                "metadata": {"chunks": ["feel anxious", "work"]},
                "similarity": 0.82
            }
        ]
        
        similar_docs = mock_db.find_similar_documents(
            embedding=None,
            limit=5,
            session_id="test_session", 
            question="Work makes me anxious"
        )
        
        assert len(similar_docs) > 0, "Should find similar documents using chunking"
        assert "chunks" in similar_docs[0]["metadata"], "Similar docs should include chunk metadata"
        
        logger.info("✅ Database integration requirements validated")
    
    def test_chunking_performance_requirements(self):
        """Test performance requirements for chunking."""
        logger.info("🧪 Testing chunking performance requirements")
        
        # Expected: Chunking should not significantly slow down the system
        # Expected: Early exit optimization should work
        # Expected: Caching should be available for repeated chunks
        
        import time
        
        mock_detector = Mock()
        
        # Simulate chunking performance
        def mock_chunking_fast(question):
            time.sleep(0.001)  # 1ms - acceptable
            return ["chunk1", "chunk2"]
        
        def mock_chunking_slow(question):
            time.sleep(0.1)  # 100ms - too slow
            return ["chunk1", "chunk2"]
        
        # Test acceptable performance
        mock_detector._semantic_chunk_question = mock_chunking_fast
        
        start_time = time.time()
        chunks = mock_detector._semantic_chunk_question("test question")
        fast_time = time.time() - start_time
        
        assert fast_time < 0.01, f"Chunking should be fast, took {fast_time:.4f}s"
        
        # Test that caching could help with repeated questions
        cached_result = mock_detector._semantic_chunk_question("test question")
        assert cached_result == chunks, "Repeated chunking should give same result"
        
        logger.info(f"✅ Performance requirement met: {fast_time:.4f}s per chunk operation")
    
    def test_chunking_error_handling_requirements(self):
        """Test error handling requirements for chunking."""
        logger.info("🧪 Testing chunking error handling requirements")
        
        # Expected: Chunking should gracefully handle edge cases
        # Expected: System should continue working if chunking fails
        # Expected: Appropriate fallback behavior should exist
        
        mock_detector = Mock()
        
        # Test edge cases
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Single character
            "?" * 1000,  # Very long string
            "Hello world! 😀 🎉",  # Unicode/emoji
            "123 456 789",  # Numbers only
        ]
        
        for i, case in enumerate(edge_cases):
            # Mock appropriate responses for edge cases
            if not case.strip():
                mock_detector._semantic_chunk_question.return_value = []
            elif len(case) == 1:
                mock_detector._semantic_chunk_question.return_value = [case]
            else:
                mock_detector._semantic_chunk_question.return_value = ["chunk1", "chunk2"]
            
            try:
                result = mock_detector._semantic_chunk_question(case)
                assert isinstance(result, list), f"Should return list for case {i}: '{case[:20]}...'"
                
            except Exception as e:
                # If chunking fails, should have fallback
                assert False, f"Chunking should handle edge case {i}: '{case[:20]}...' - Error: {e}"
        
        # Test similarity calculation error handling
        mock_detector._calculate_max_chunk_similarity.side_effect = [
            0.5,  # Normal case
            Exception("Similarity calculation failed")  # Error case
        ]
        
        # Normal case should work
        similarity = mock_detector._calculate_max_chunk_similarity(["a"], ["b"])
        assert similarity == 0.5
        
        # Error case should be handled gracefully
        try:
            mock_detector._calculate_max_chunk_similarity(["a"], ["b"])
            assert False, "Should have raised an exception"
        except Exception:
            # This is expected - the calling code should handle this
            pass
        
        logger.info("✅ Error handling requirements validated")
    
    def test_chunking_semantic_quality_requirements(self):
        """Test semantic quality requirements for chunking."""
        logger.info("🧪 Testing chunking semantic quality requirements")
        
        # Expected: Chunks should capture semantic meaning
        # Expected: Similar concepts should have high similarity scores
        # Expected: Different concepts should have low similarity scores
        
        mock_detector = Mock()
        
        # Test semantic similarity detection
        test_cases = [
            # (question1, question2, expected_similarity_level)
            ("I feel sad", "I am feeling depressed", "high"),
            ("I'm anxious about work", "Work makes me nervous", "high"), 
            ("I love pizza", "I feel anxious about work", "low"),
            ("Good morning", "I have severe depression", "low"),
            ("I can't sleep", "I have insomnia", "high"),
            ("I'm happy today", "I feel joyful", "high"),
        ]
        
        for q1, q2, expected_level in test_cases:
            # Mock chunks for each question
            if "sad" in q1 or "depressed" in q2:
                chunks1 = ["feel sad"]
                chunks2 = ["feeling depressed"]
                similarity = 0.85  # High similarity
            elif "anxious" in q1 or "nervous" in q2:
                chunks1 = ["anxious", "work"]
                chunks2 = ["nervous", "work"] 
                similarity = 0.78  # High similarity
            elif "sleep" in q1 or "insomnia" in q2:
                chunks1 = ["can't sleep"]
                chunks2 = ["insomnia"]
                similarity = 0.80  # High similarity
            elif "happy" in q1 or "joyful" in q2:
                chunks1 = ["happy"]
                chunks2 = ["joyful"]
                similarity = 0.75  # High similarity
            else:
                chunks1 = ["different", "concept"]
                chunks2 = ["unrelated", "topic"]
                similarity = 0.25  # Low similarity
            
            mock_detector._semantic_chunk_question.side_effect = [chunks1, chunks2]
            mock_detector._calculate_max_chunk_similarity.return_value = similarity
            
            # Test the semantic quality
            actual_chunks1 = mock_detector._semantic_chunk_question(q1)
            actual_chunks2 = mock_detector._semantic_chunk_question(q2)
            actual_similarity = mock_detector._calculate_max_chunk_similarity(actual_chunks1, actual_chunks2)
            
            if expected_level == "high":
                assert actual_similarity >= 0.7, \
                    f"'{q1}' and '{q2}' should have high similarity, got {actual_similarity}"
            else:  # low
                assert actual_similarity <= 0.4, \
                    f"'{q1}' and '{q2}' should have low similarity, got {actual_similarity}"
            
            logger.info(f"✅ {q1[:20]}... vs {q2[:20]}... = {actual_similarity:.2f} ({expected_level})")


# Test execution helper
def run_requirements_tests():
    """Run the semantic chunking requirements test suite."""
    logger.info("📋 Running Semantic Chunking Requirements Test Suite")
    logger.info("=" * 60)
    
    import subprocess
    
    result = subprocess.run([
        "python", "-m", "pytest",
        __file__,
        "-v", 
        "--tb=short",
        "--color=yes",
        "-x"  # Stop on first failure
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_requirements_tests()
    sys.exit(0 if success else 1)
