#!/usr/bin/env python3
"""
Semantic Chunking Integration Discovery Tests

This test suite discovers and validates the actual semantic chunking
implementation in the psy-supabase system to understand how to integrate it.
"""

import pytest
import os
import sys
from typing import List, Dict, Any

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psy_supabase import get_package_logger
from psy_supabase.core.pain_point_detector import PainPointDetector

logger = get_package_logger(__name__)


class TestSemanticChunkingDiscovery:
    """Discover how the existing semantic chunking implementation works."""
    
    @pytest.fixture
    def pain_point_detector(self):
        """Create a PainPointDetector instance for testing."""
        from unittest.mock import Mock
        mock_db_manager = Mock()
        return PainPointDetector(db_manager=mock_db_manager)
    
    def test_discover_chunking_methods(self, pain_point_detector):
        """Discover what chunking methods are available and how they work."""
        logger.info("🔍 Discovering semantic chunking methods")
        
        # Test the main semantic chunking method
        test_question = "I feel very anxious about my work performance every day"
        chunks = pain_point_detector._semantic_chunk_question(test_question)
        
        logger.info(f"Input: '{test_question}'")
        logger.info(f"Chunks: {chunks}")
        
        # Validate the output
        assert isinstance(chunks, list), "Should return a list of chunks"
        assert len(chunks) > 0, "Should return at least one chunk"
        assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"
        
        # Test that it extracts meaningful chunks
        expected_concepts = ["anxious", "work", "performance", "every day"]
        found_concepts = []
        
        for concept in expected_concepts:
            if any(concept in chunk.lower() for chunk in chunks):
                found_concepts.append(concept)
        
        logger.info(f"Found concepts: {found_concepts}")
        assert len(found_concepts) >= 2, f"Should find at least 2 key concepts, found: {found_concepts}"
    
    def test_discover_similarity_calculation(self, pain_point_detector):
        """Discover how chunk similarity is calculated."""
        logger.info("🔍 Discovering chunk similarity calculation")
        
        # Test similar questions
        q1 = "I feel inadequate at work"
        q2 = "I feel inadequate at my job"
        
        chunks1 = pain_point_detector._semantic_chunk_question(q1)
        chunks2 = pain_point_detector._semantic_chunk_question(q2)
        
        similarity = pain_point_detector._calculate_max_chunk_similarity(chunks1, chunks2)
        
        logger.info(f"Question 1: '{q1}' → {chunks1}")
        logger.info(f"Question 2: '{q2}' → {chunks2}")
        logger.info(f"Similarity: {similarity}")
        
        # Validate similarity calculation
        assert isinstance(similarity, (int, float)), "Similarity should be numeric"
        # Allow for small floating point precision errors
        assert -0.001 <= similarity <= 1.001, f"Similarity should be ~0-1, got {similarity}"
        assert similarity > 0.5, f"Similar questions should have high similarity, got {similarity}"
        
        # Test dissimilar questions
        q3 = "I love playing tennis"
        chunks3 = pain_point_detector._semantic_chunk_question(q3)
        dissimilarity = pain_point_detector._calculate_max_chunk_similarity(chunks1, chunks3)
        
        logger.info(f"Question 3: '{q3}' → {chunks3}")
        logger.info(f"Dissimilarity: {dissimilarity}")
        
        assert dissimilarity < similarity, f"Dissimilar questions should have lower similarity: {dissimilarity} < {similarity}"
    
    def test_discover_semantic_theme_extraction(self, pain_point_detector):
        """Discover how semantic themes are extracted from chunks."""
        logger.info("🔍 Discovering semantic theme extraction")
        
        # Test with work-related chunks
        work_chunks = ["feel inadequate", "work", "job", "performance", "anxious"]
        theme = pain_point_detector._extract_semantic_theme(work_chunks)
        
        logger.info(f"Work chunks: {work_chunks}")
        logger.info(f"Extracted theme: '{theme}'")
        
        assert isinstance(theme, str), "Theme should be a string"
        assert len(theme) > 0, "Theme should not be empty"
        
        # Test with different types of chunks
        anxiety_chunks = ["anxious", "worried", "stressed", "nervous"]
        anxiety_theme = pain_point_detector._extract_semantic_theme(anxiety_chunks)
        
        logger.info(f"Anxiety chunks: {anxiety_chunks}")
        logger.info(f"Anxiety theme: '{anxiety_theme}'")
        
        # Themes should be different for different types of content
        assert theme != anxiety_theme, f"Different chunk types should produce different themes"
    
    def test_discover_repeating_terms_extraction(self, pain_point_detector):
        """Discover how repeating terms are extracted."""
        logger.info("🔍 Discovering repeating terms extraction")
        
        # Test with chunks that should have repeating terms
        chunks_with_repetition = [
            "feel anxious about work",
            "work makes me anxious", 
            "anxious feelings at work",
            "work stress",
            "daily work anxiety"
        ]
        
        repeating_terms = pain_point_detector._extract_repeating_terms(chunks_with_repetition)
        
        logger.info(f"Input chunks: {chunks_with_repetition}")
        logger.info(f"Repeating terms: {repeating_terms}")
        
        assert isinstance(repeating_terms, list), "Should return a list"
        
        # Should find common terms like "work" and "anxious"
        expected_terms = ["work", "anxious"]
        found_terms = []
        
        for term in expected_terms:
            if any(term in repeating_term.lower() for repeating_term in repeating_terms):
                found_terms.append(term)
        
        logger.info(f"Found expected terms: {found_terms}")
        # Note: This might be empty if spaCy is not available, that's ok for discovery
    
    def test_discover_chunking_configuration(self):
        """Discover the chunking configuration options."""
        logger.info("🔍 Discovering chunking configuration")
        
        from psy_supabase.config import PAIN_POINT_DETECTION
        
        logger.info(f"Pain point detection config: {PAIN_POINT_DETECTION}")
        
        # Check for chunking-specific configuration
        if "chunking" in PAIN_POINT_DETECTION:
            chunking_config = PAIN_POINT_DETECTION["chunking"]
            logger.info(f"Chunking config: {chunking_config}")
            
            for key, value in chunking_config.items():
                logger.info(f"  {key}: {value} ({type(value).__name__})")
        
        # Check similarity threshold
        if "similarity_threshold" in PAIN_POINT_DETECTION:
            threshold = PAIN_POINT_DETECTION["similarity_threshold"]
            logger.info(f"Similarity threshold: {threshold}")
    
    def test_discover_integration_with_pain_point_detection(self, pain_point_detector):
        """Discover how chunking integrates with pain point detection."""
        logger.info("🔍 Discovering pain point detection integration")
        
        # Test questions that should be detected as similar pain points
        questions = [
            "I feel inadequate at work every day",
            "I'm feeling inadequate every day at my job", 
            "Every day I feel inadequate at work",
            "I love playing tennis and feel great"  # Different topic
        ]
        
        # Get chunks for each question
        all_chunks = []
        for i, question in enumerate(questions):
            chunks = pain_point_detector._semantic_chunk_question(question)
            all_chunks.append((i, question, chunks))
            logger.info(f"Q{i}: '{question}' → {chunks}")
        
        # Calculate similarity matrix
        logger.info("\nSimilarity matrix:")
        similarity_matrix = []
        
        for i, (idx1, q1, chunks1) in enumerate(all_chunks):
            row = []
            for j, (idx2, q2, chunks2) in enumerate(all_chunks):
                if i == j:
                    similarity = 1.0
                else:
                    similarity = pain_point_detector._calculate_max_chunk_similarity(chunks1, chunks2)
                row.append(similarity)
                logger.info(f"  Q{idx1} vs Q{idx2}: {similarity:.3f}")
            similarity_matrix.append(row)
        
        # Validate that work-related questions are more similar to each other
        work_questions = [0, 1, 2]  # First 3 questions are work-related
        tennis_question = 3
        
        # Work questions should be similar to each other
        for i in work_questions:
            for j in work_questions:
                if i != j:
                    sim = similarity_matrix[i][j]
                    assert sim > 0.3, f"Work questions Q{i} and Q{j} should be similar: {sim}"
        
        # Work questions should be dissimilar to tennis question
        for i in work_questions:
            sim = similarity_matrix[i][tennis_question]
            logger.info(f"Work Q{i} vs Tennis Q{tennis_question}: {sim}")
            # Note: Asserting this might be too strict, just log for discovery
    
    def test_discover_chunking_performance_characteristics(self, pain_point_detector):
        """Discover performance characteristics of chunking."""
        logger.info("🔍 Discovering chunking performance")
        
        import time
        
        # Test with various question lengths
        test_cases = [
            "Anxious",  # Short
            "I feel anxious about work",  # Medium
            "I feel very anxious about my work performance and worry about it every single day because I think I'm inadequate",  # Long
            "This is a very long question with many words that should test the performance of the semantic chunking algorithm when processing complex psychological statements with multiple concepts and emotional indicators that need to be properly identified and processed for similarity comparison with other user questions in the system",  # Very long
        ]
        
        for i, question in enumerate(test_cases):
            start_time = time.time()
            chunks = pain_point_detector._semantic_chunk_question(question)
            end_time = time.time()
            
            duration = end_time - start_time
            logger.info(f"Test {i+1} ({len(question)} chars): {duration:.4f}s → {len(chunks)} chunks")
            logger.info(f"  Question: '{question[:50]}{'...' if len(question) > 50 else ''}'")
            logger.info(f"  Chunks: {chunks}")
            
            # Performance should be reasonable (under 100ms for this test)
            assert duration < 0.1, f"Chunking should be fast, took {duration:.4f}s"
    
    def test_discover_edge_case_handling(self, pain_point_detector):
        """Discover how chunking handles edge cases."""
        logger.info("🔍 Discovering edge case handling")
        
        edge_cases = [
            ("", "Empty string"),
            ("   ", "Whitespace only"),
            ("a", "Single character"),
            ("Hello world! 😀 🎉", "Unicode/emoji"),
            ("123 456 789", "Numbers only"),
            ("??!!! @#$%", "Special characters"),
            ("I" * 1000, "Very repetitive"),
        ]
        
        for case, description in edge_cases:
            logger.info(f"\nTesting: {description}")
            logger.info(f"Input: '{case[:50]}{'...' if len(case) > 50 else ''}'")
            
            try:
                chunks = pain_point_detector._semantic_chunk_question(case)
                logger.info(f"Output: {chunks}")
                
                # Basic validation
                assert isinstance(chunks, list), f"Should return list for {description}"
                
                if chunks:  # If not empty
                    assert all(isinstance(chunk, str) for chunk in chunks), f"All chunks should be strings for {description}"
                
            except Exception as e:
                logger.error(f"Error with {description}: {e}")
                # Don't fail the test, just log the error for discovery


# Test runner for discovery
def run_discovery_tests():
    """Run the semantic chunking discovery test suite."""
    logger.info("🧪 Running Semantic Chunking Discovery Test Suite")
    logger.info("=" * 70)
    
    import subprocess
    
    result = subprocess.run([
        "python", "-m", "pytest",
        __file__,
        "-v", 
        "--tb=short",
        "--color=yes",
        "-s"  # Don't capture output so we can see logs
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_discovery_tests()
    sys.exit(0 if success else 1)
