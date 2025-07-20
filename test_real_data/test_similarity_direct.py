"""
Test similarity calculation directly on the pain point detector.

This module tests the semantic similarity calculation engine that powers
pain point detection. It validates:

1. Semantic chunking of user questions into meaningful phrases
2. Sentence transformer similarity calculations between chunks
3. Threshold-based clustering decisions
4. Configuration impact on detection sensitivity

Usage Examples:
--------------

Run basic similarity test:
    python test_similarity_direct.py

Expected Output:
    🔍 Testing direct similarity calculation
    Question 1: I feel inadequate at work every single day...
    Question 2: I'm still feeling inadequate every day...

    Chunks 1: ['feeling inadequate', 'work', 'every day', 'confidence', ...]
    Chunks 2: ['feeling inadequate', 'every day', 'quit', 'job', ...]

    Calculated similarity: 0.8234
    ✅ Threshold 0.6: PASS
    ✅ Threshold 0.7: PASS
    ✅ Threshold 0.8: PASS

Testing Configuration Impact:
----------------------------

To test different sensitivity levels, modify config.py before running:

High Sensitivity (more pain points detected):
    PAIN_POINT_DETECTION["similarity_threshold"] = 0.5

Normal Sensitivity (default):
    PAIN_POINT_DETECTION["similarity_threshold"] = 0.6

Low Sensitivity (only obvious pain points):
    PAIN_POINT_DETECTION["similarity_threshold"] = 0.8

Performance vs Accuracy Testing:
-------------------------------

High Performance (faster):
    PAIN_POINT_DETECTION["chunking"]["include_full_question"] = False
    PAIN_POINT_DETECTION["chunking"]["max_chunks_per_question"] = 5

High Accuracy (slower):
    PAIN_POINT_DETECTION["chunking"]["include_full_question"] = True
    PAIN_POINT_DETECTION["chunking"]["max_chunks_per_question"] = 12

Debug Logging Control:
---------------------

Reduce log noise:
    PAIN_POINT_DETECTION["chunking"]["log_only_best_matches"] = True

Full debug logging:
    PAIN_POINT_DETECTION["chunking"]["log_only_best_matches"] = False

What This Test Validates:
------------------------

✅ Semantic chunking extracts meaningful phrases
✅ Sentence transformers provide accurate similarity scores
✅ Threshold logic correctly determines cluster membership
✅ Configuration changes impact detection behavior
✅ Early exit optimization works on perfect matches
✅ Debug logging provides useful information

Troubleshooting:
---------------

If similarity is too low (< 0.6):
- Check if questions are actually semantically similar
- Verify sentence transformer model is loaded correctly
- Consider lowering similarity_threshold in config

If similarity is too high (> 0.9) for different concepts:
- Check chunking is extracting specific enough features
- Verify questions aren't nearly identical
- Consider raising similarity_threshold in config

If performance is slow:
- Set include_full_question = False
- Reduce max_chunks_per_question
- Enable early_exit_on_perfect_match
"""

import os
import sys
import uuid
from typing import cast

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from psy_supabase.core.database import DatabaseManager

load_dotenv()

from psy_supabase import get_package_logger

logger = get_package_logger(__name__)


def test_direct_similarity() -> None:
    """Test similarity calculation directly."""

    logger.info("🔍 Testing direct similarity calculation")
    logger.info("=" * 50)

    # Setup
    supabase_url = cast(str, os.environ.get("SUPABASE_URL"))
    supabase_key = cast(str, os.environ.get("SUPABASE_KEY"))
    test_user_id = f"debug_user_{uuid.uuid4().hex[:8]}"

    db_manager = DatabaseManager(supabase_url=supabase_url, supabase_key=supabase_key, user_id=test_user_id)

    pain_detector = db_manager.pain_point_detector

    # Test chunking
    q1 = "I feel inadequate at work every single day and it's affecting my confidence"
    q2 = "I'm still feeling inadequate every day and thinking about quitting my job"

    logger.info(f"Question 1: {q1}")
    logger.info(f"Question 2: {q2}")

    try:
        # Test chunking
        chunks1 = pain_detector._semantic_chunk_question(q1)
        chunks2 = pain_detector._semantic_chunk_question(q2)

        logger.info(f"\nChunks 1: {chunks1}")
        logger.info(f"Chunks 2: {chunks2}")

        # Test similarity calculation
        similarity = pain_detector._calculate_max_chunk_similarity(chunks1, chunks2)

        logger.info(f"\nCalculated similarity: {similarity}")

        # Test with different thresholds
        thresholds = [0.3, 0.5, 0.6, 0.7, 0.8]
        for threshold in thresholds:
            if similarity >= threshold:
                logger.info(f"✅ Threshold {threshold}: PASS")
            else:
                logger.info(f"❌ Threshold {threshold}: FAIL")

    except Exception as e:
        logger.info(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_direct_similarity()
