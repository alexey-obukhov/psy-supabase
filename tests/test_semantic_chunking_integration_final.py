#!/usr/bin/env python3
"""
Semantic Chunking Integration Final Test

This test demonstrates and validates the semantic chunking integration
with actual working examples based on our discoveries.
"""

import pytest
import os
import sys
from unittest.mock import Mock
from typing import List, Dict, Any

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psy_supabase import get_package_logger
from psy_supabase.core.pain_point_detector import PainPointDetector
from psy_supabase.config import PAIN_POINT_DETECTION

logger = get_package_logger(__name__)


class TestSemanticChunkingIntegrationFinal:
    """Final integration test demonstrating how semantic chunking works in practice."""
    
    @pytest.fixture
    def pain_point_detector(self):
        """Create a PainPointDetector instance for testing."""
        mock_db_manager = Mock()
        return PainPointDetector(db_manager=mock_db_manager)
    
    def test_semantic_chunking_real_world_scenarios(self, pain_point_detector):
        """Test semantic chunking with real psychological scenarios."""
        logger.info("🧪 Testing Real-World Semantic Chunking Scenarios")
        
        # Test Case 1: Work-related anxiety variations
        work_anxiety_variations = [
            "I feel very anxious about my work performance",
            "My job performance makes me nervous",
            "I'm worried about how I'm doing at work",
            "Work stress is overwhelming me"
        ]
        
        logger.info("\n📊 Work Anxiety Variations:")
        all_chunks = []
        for i, question in enumerate(work_anxiety_variations):
            chunks = pain_point_detector._semantic_chunk_question(question)
            all_chunks.append((i, question, chunks))
            logger.info(f"  Q{i+1}: '{question}'")
            logger.info(f"       → {chunks}")
        
        # Test similarity between work-related questions
        logger.info("\n🔄 Similarity Analysis:")
        for i in range(len(all_chunks)):
            for j in range(i + 1, len(all_chunks)):
                chunks1 = all_chunks[i][2]
                chunks2 = all_chunks[j][2]
                similarity = pain_point_detector._calculate_max_chunk_similarity(chunks1, chunks2)
                
                logger.info(f"  Q{i+1} vs Q{j+1}: {similarity:.3f}")
                
                # All work-related questions should have reasonable similarity
                assert similarity >= 0.3, f"Work-related questions should be similar: {similarity}"
        
        # Test Case 2: Different topics should have low similarity
        different_topic = "I love playing tennis on weekends"
        different_chunks = pain_point_detector._semantic_chunk_question(different_topic)
        
        logger.info(f"\n🎾 Different Topic: '{different_topic}'")
        logger.info(f"     → {different_chunks}")
        
        for i, (_, question, chunks) in enumerate(all_chunks):
            similarity = pain_point_detector._calculate_max_chunk_similarity(chunks, different_chunks)
            logger.info(f"  Work Q{i+1} vs Tennis: {similarity:.3f}")
            
            # Different topics should have lower similarity
            assert similarity < 0.8, f"Different topics should have lower similarity: {similarity}"
    
    def test_chunking_configuration_integration(self):
        """Test that chunking configuration is properly integrated."""
        logger.info("🧪 Testing Chunking Configuration Integration")
        
        config = PAIN_POINT_DETECTION
        logger.info(f"Configuration loaded: {config}")
        
        # Validate all expected configuration options
        assert 'chunking' in config, "Chunking configuration should be present"
        assert 'similarity_threshold' in config, "Similarity threshold should be configured"
        
        chunking_config = config['chunking']
        expected_options = [
            'include_full_question',
            'min_chunk_length', 
            'max_chunks_per_question',
            'remove_redundant_chunks',
            'extract_noun_phrases',
            'extract_emotional_phrases',
            'extract_temporal_phrases',
            'extract_action_phrases',
            'early_exit_on_perfect_match',
            'log_only_best_matches'
        ]
        
        for option in expected_options:
            assert option in chunking_config, f"Configuration should include {option}"
            logger.info(f"  ✅ {option}: {chunking_config[option]}")
        
        # Validate similarity threshold
        threshold = config['similarity_threshold']
        assert 0.0 <= threshold <= 1.0, f"Similarity threshold should be 0-1: {threshold}"
        logger.info(f"  ✅ similarity_threshold: {threshold}")
    
    def test_chunking_pain_point_detection_integration(self, pain_point_detector):
        """Test how chunking integrates with pain point detection workflow."""
        logger.info("🧪 Testing Pain Point Detection Integration")
        
        # Simulate a sequence of related questions over time
        session_questions = [
            "I feel inadequate at work every day",
            "My work performance is terrible", 
            "I can't handle the pressure at my job",
            "Every day at work I feel like a failure",
            "I'm thinking about quitting because I'm so bad at my job"
        ]
        
        logger.info("📝 Session Questions Analysis:")
        pain_point_data = []
        
        for i, question in enumerate(session_questions):
            # Step 1: Extract chunks
            chunks = pain_point_detector._semantic_chunk_question(question)
            
            # Step 2: Calculate similarities with previous questions
            similarities = []
            for j in range(i):
                prev_chunks = pain_point_data[j]['chunks']
                similarity = pain_point_detector._calculate_max_chunk_similarity(chunks, prev_chunks)
                similarities.append((j, similarity))
            
            # Step 3: Store pain point data
            pain_point_entry = {
                'question': question,
                'chunks': chunks,
                'similarities': similarities,
                'max_similarity': max([s[1] for s in similarities]) if similarities else 0.0
            }
            pain_point_data.append(pain_point_entry)
            
            logger.info(f"\n  Q{i+1}: '{question}'")
            logger.info(f"       Chunks: {chunks}")
            if similarities:
                logger.info(f"       Max similarity: {pain_point_entry['max_similarity']:.3f}")
                
                # Check if this constitutes a recurring pain point
                if pain_point_entry['max_similarity'] >= PAIN_POINT_DETECTION['similarity_threshold']:
                    logger.info(f"       🚨 PAIN POINT DETECTED (similarity >= {PAIN_POINT_DETECTION['similarity_threshold']})")
        
        # Validate that related questions were properly identified
        similar_pairs = []
        for entry in pain_point_data:
            if entry['max_similarity'] >= PAIN_POINT_DETECTION['similarity_threshold']:
                similar_pairs.append(entry)
        
        assert len(similar_pairs) >= 2, f"Should detect multiple similar questions: {len(similar_pairs)}"
        logger.info(f"\n✅ Detected {len(similar_pairs)} questions with recurring themes")
    
    def test_chunking_performance_and_optimization(self, pain_point_detector):
        """Test chunking performance optimizations work correctly."""
        logger.info("🧪 Testing Chunking Performance Optimizations")
        
        # Test early exit optimization
        identical_questions = [
            "I feel anxious about work",
            "I feel anxious about work"  # Identical
        ]
        
        chunks1 = pain_point_detector._semantic_chunk_question(identical_questions[0])
        chunks2 = pain_point_detector._semantic_chunk_question(identical_questions[1])
        
        logger.info(f"Q1: '{identical_questions[0]}' → {chunks1}")
        logger.info(f"Q2: '{identical_questions[1]}' → {chunks2}")
        
        # Should get perfect match due to early exit optimization
        similarity = pain_point_detector._calculate_max_chunk_similarity(chunks1, chunks2)
        logger.info(f"Similarity: {similarity}")
        
        # Allow for tiny floating point differences
        assert abs(similarity - 1.0) < 0.001, f"Identical questions should have ~1.0 similarity: {similarity}"
        
        # Test that chunking respects max_chunks_per_question
        max_chunks = PAIN_POINT_DETECTION['chunking']['max_chunks_per_question']
        
        long_question = "I feel very anxious and worried and stressed and overwhelmed about my terrible awful work performance and job responsibilities every single day"
        chunks = pain_point_detector._semantic_chunk_question(long_question)
        
        logger.info(f"\nLong question: '{long_question}'")
        logger.info(f"Chunks ({len(chunks)}): {chunks}")
        logger.info(f"Max allowed: {max_chunks}")
        
        assert len(chunks) <= max_chunks, f"Should not exceed max chunks: {len(chunks)} > {max_chunks}"
    
    def test_integration_with_database_workflow(self, pain_point_detector):
        """Test how chunking integrates with database storage workflow."""
        logger.info("🧪 Testing Database Integration Workflow")
        
        # Mock database manager
        mock_db = pain_point_detector.db_manager
        
        # Simulate interaction data with chunking metadata
        test_question = "I'm struggling with anxiety at work every day"
        chunks = pain_point_detector._semantic_chunk_question(test_question)
        
        # Create interaction data that would be stored
        interaction_data = {
            "question": test_question,
            "answer": "I understand you're experiencing workplace anxiety...",
            "metadata": {
                "semantic_chunks": chunks,
                "chunk_analysis": {
                    "noun_phrases": [c for c in chunks if len(c.split()) > 1],
                    "key_terms": [c for c in chunks if len(c.split()) == 1],
                    "full_question": test_question.lower()
                },
                "pain_point_analysis": {
                    "similarity_threshold": PAIN_POINT_DETECTION['similarity_threshold'],
                    "chunking_config": PAIN_POINT_DETECTION['chunking']
                }
            }
        }
        
        logger.info(f"Question: {interaction_data['question']}")
        logger.info(f"Chunks: {interaction_data['metadata']['semantic_chunks']}")
        logger.info(f"Analysis: {interaction_data['metadata']['chunk_analysis']}")
        
        # Validate metadata structure
        assert 'semantic_chunks' in interaction_data['metadata']
        assert 'chunk_analysis' in interaction_data['metadata']
        assert 'pain_point_analysis' in interaction_data['metadata']
        
        # Test vector similarity query preparation
        # This would be used for finding similar previous interactions
        query_chunks = chunks
        similarity_threshold = PAIN_POINT_DETECTION['similarity_threshold']
        
        logger.info(f"\nVector Query Preparation:")
        logger.info(f"  Query chunks: {query_chunks}")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        
        # Mock finding similar interactions
        mock_similar_interactions = [
            {
                "question": "Work makes me anxious every day",
                "metadata": {"semantic_chunks": ["work", "anxious", "every day"]},
                "similarity_score": 0.85
            },
            {
                "question": "I love going to concerts",
                "metadata": {"semantic_chunks": ["love", "concerts"]}, 
                "similarity_score": 0.15
            }
        ]
        
        # Filter by similarity threshold
        relevant_interactions = [
            interaction for interaction in mock_similar_interactions
            if interaction['similarity_score'] >= similarity_threshold
        ]
        
        logger.info(f"  Similar interactions found: {len(relevant_interactions)}")
        for interaction in relevant_interactions:
            logger.info(f"    '{interaction['question']}' (score: {interaction['similarity_score']})")
        
        assert len(relevant_interactions) >= 1, "Should find at least one similar interaction"
    
    def test_end_to_end_chunking_workflow(self, pain_point_detector):
        """Test the complete end-to-end chunking workflow."""
        logger.info("🧪 Testing End-to-End Chunking Workflow")
        
        # Simulate a complete user session with chunking-enhanced pain point detection
        session_id = "test_session_123"
        user_questions = [
            "Hi, I'm feeling stressed lately",
            "I'm having trouble with my work performance", 
            "Every day at work I feel like I'm not good enough",
            "I think I might be inadequate at my job",
            "Should I quit because I'm so bad at work?"
        ]
        
        session_data = {
            "session_id": session_id,
            "interactions": [],
            "pain_points_detected": [],
            "chunking_analytics": {
                "total_chunks": 0,
                "unique_chunks": set(),
                "recurring_themes": {}
            }
        }
        
        logger.info(f"Session: {session_id}")
        logger.info("Processing questions:")
        
        for i, question in enumerate(user_questions):
            # Step 1: Extract semantic chunks
            chunks = pain_point_detector._semantic_chunk_question(question)
            
            # Step 2: Check similarity with previous interactions
            max_similarity = 0.0
            most_similar_interaction = None
            
            for prev_interaction in session_data["interactions"]:
                prev_chunks = prev_interaction["chunks"]
                similarity = pain_point_detector._calculate_max_chunk_similarity(chunks, prev_chunks)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_interaction = prev_interaction
            
            # Step 3: Determine if this is a recurring pain point
            is_pain_point = max_similarity >= PAIN_POINT_DETECTION['similarity_threshold']
            
            # Step 4: Store interaction
            interaction = {
                "question_id": i + 1,
                "question": question,
                "chunks": chunks,
                "max_similarity": max_similarity,
                "is_pain_point": is_pain_point,
                "similar_to": most_similar_interaction["question_id"] if most_similar_interaction else None
            }
            
            session_data["interactions"].append(interaction)
            
            # Step 5: Update analytics
            session_data["chunking_analytics"]["total_chunks"] += len(chunks)
            session_data["chunking_analytics"]["unique_chunks"].update(chunks)
            
            # Track recurring themes
            for chunk in chunks:
                if chunk not in session_data["chunking_analytics"]["recurring_themes"]:
                    session_data["chunking_analytics"]["recurring_themes"][chunk] = 0
                session_data["chunking_analytics"]["recurring_themes"][chunk] += 1
            
            # Step 6: Record pain point if detected
            if is_pain_point:
                pain_point = {
                    "theme": pain_point_detector._extract_semantic_theme(chunks),
                    "questions": [interaction["question"], most_similar_interaction["question"]],
                    "similarity_score": max_similarity
                }
                session_data["pain_points_detected"].append(pain_point)
            
            # Log results
            logger.info(f"\n  Q{i+1}: '{question}'")
            logger.info(f"       Chunks: {chunks}")
            logger.info(f"       Max similarity: {max_similarity:.3f}")
            if is_pain_point:
                logger.info(f"       🚨 PAIN POINT: Similar to Q{interaction['similar_to']}")
        
        # Final analytics
        analytics = session_data["chunking_analytics"]
        analytics["unique_chunks"] = list(analytics["unique_chunks"])
        
        logger.info(f"\n📊 Session Analytics:")
        logger.info(f"  Total chunks: {analytics['total_chunks']}")
        logger.info(f"  Unique chunks: {len(analytics['unique_chunks'])}")
        logger.info(f"  Pain points detected: {len(session_data['pain_points_detected'])}")
        
        # Show most frequent themes
        frequent_themes = sorted(
            analytics["recurring_themes"].items(),
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        logger.info(f"  Top themes:")
        for theme, count in frequent_themes:
            logger.info(f"    '{theme}': {count} times")
        
        # Validate workflow results
        assert len(session_data["interactions"]) == len(user_questions)
        assert len(session_data["pain_points_detected"]) >= 1, "Should detect at least one pain point"
        assert analytics["total_chunks"] > 0, "Should generate chunks"
        assert len(analytics["unique_chunks"]) > 0, "Should have unique chunks"
        
        logger.info(f"\n✅ End-to-end workflow completed successfully!")


# Test runner
def run_integration_tests():
    """Run the semantic chunking integration test suite."""
    logger.info("🚀 Running Semantic Chunking Integration Tests")
    logger.info("=" * 70)
    
    import subprocess
    
    result = subprocess.run([
        "python", "-m", "pytest",
        __file__,
        "-v", 
        "--tb=short",
        "--color=yes",
        "-s"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
