"""
Test pain point integration with real database using the working pattern from test_pain_point_detection_detailed.py
"""

import json
import os
import sys
import uuid
import time
from typing import Any, Dict, cast

from psy_supabase import get_package_logger

# Add project to path for imports (exactly like your working code)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logger = get_package_logger(__name__)

import nltk
try:
    logger.info("📝 Ensuring NLTK resources are available...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("✅ NLTK resources ready")
except Exception as e:
    logger.info(f"⚠️ NLTK download warning: {e}")

from psy_supabase.core.database import DatabaseManager
from psy_supabase.utilities.common import is_github_actions
from psy_supabase.utilities.utils import cleanup_memory

# Load environment variables (exactly like your working code)
if not is_github_actions():
    from dotenv import load_dotenv
    load_dotenv()

# Require Supabase credentials (exactly like your working code)
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.critical("Missing Supabase credentials")
    sys.exit(1)

def test_pain_point_integration_real():
    """Test pain point integration using the exact working pattern."""

    logger.info("🔗 Testing pain point integration with real database")
    logger.info("=" * 60)

    try:
        # Create unique test identifiers (exactly like your working code)
        test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        test_session_id = f"pp_integration_{uuid.uuid4().hex[:10]}"
        logger.info(f"Test session ID: {test_session_id}")

        supabase_url_test = cast(str, supabase_url)
        supabase_key_test = cast(str, supabase_key)

        # Initialize core components (exactly like your working code)
        logger.info(f"📝 Creating database manager with user_id: {test_user_id}")
        db_manager = DatabaseManager(
            supabase_url=supabase_url_test,
            supabase_key=supabase_key_test,
            user_id=test_user_id
        )

        logger.info(f"✅ Database manager created with schema: {db_manager.schema_name}")

        # Setup test environment (exactly like your working code)
        logger.info("📝 Setting up test environment...")
        db_manager.create_user_schema_sync()
        logger.info("✅ Test environment setup complete")

        logger.info(f"📝 Testing with session: {test_session_id}")

        # First interaction - should NOT store pain points (first occurrence)
        logger.info("\n📝 Step 1: Adding first interaction (no pain points expected)")

        first_interaction = {
            "question": "I feel inadequate at work every single day and it's affecting my confidence",
            "answer": "I understand that feeling inadequate at work can be very challenging and emotionally draining. These feelings of self-doubt can indeed impact your confidence significantly.",
            "context": "workplace confidence discussion",
            "metadata": {"test": True, "step": 1}
        }

        result1 = db_manager.add_interaction(first_interaction, test_session_id)
        logger.info(f"   First interaction result: {result1}")

        if result1.get('success'):
            logger.info("✅ First interaction added successfully")

            # Add delay to force temporal separation
            logger.info("📝 Adding delay to simulate temporal separation...")
            time.sleep(3)  # 3 second delay

            # Second interaction - should store pain points (recurrence detected)
            logger.info("\n📝 Step 2: Adding second interaction (pain points expected due to recurrence)")

            second_interaction = {
                "question": "I'm still feeling inadequate every day and thinking about quitting my job",
                "answer": "It sounds like these feelings of inadequacy are persistent and now affecting your career decisions. Let's explore some strategies to address these recurring thoughts.",
                "context": "career decision discussion",
                "metadata": {"test": True, "step": 2}
            }

            result2 = db_manager.add_interaction(second_interaction, test_session_id)
            logger.info(f"   Second interaction result: {result2}")

            if result2.get('success'):
                logger.info("✅ Second interaction added successfully")

                # Test temporal grouping debugging first
                logger.info("\n📝 Step 2.5: Debugging temporal grouping issue")

                # Get the raw history to check timestamps
                history = db_manager.get_conversation_history(test_session_id)
                logger.info(f"   Raw interaction timestamps:")
                for i, interaction in enumerate(history):
                    timestamp = interaction.get('created_at', 'NO_TIMESTAMP')
                    question = interaction.get('question', '')[:50]
                    logger.info(f"     {i+1}. {timestamp} - {question}...")

                # Test the temporal grouping directly
                pain_detector = db_manager.pain_point_detector

                try:
                    temporal_groups = pain_detector._group_interactions_by_time(history, 30)  # 30 days window
                    logger.info(f"   Temporal groups created: {len(temporal_groups)}")

                    for i, group in enumerate(temporal_groups):
                        logger.info(f"     Group {i}: {group.get('start_time', 'NO_START')} to {group.get('end_time', 'NO_END')}")
                        logger.info(f"       Interactions: {len(group.get('interactions', []))}")

                except Exception as temporal_e:
                    logger.info(f"   ❌ Temporal grouping failed: {temporal_e}")

                # Test with very small time window to force separate groups
                logger.info("\n   Testing with very small time window...")
                try:
                    ultra_small_result = db_manager.detect_pain_points(
                        session_id=test_session_id,
                        threshold=0.3,  # Lower threshold
                        min_occurrences=2,
                        time_window_days=0.00001  # Very small window
                    )

                    logger.info(f"   Ultra-small time window result:")
                    logger.info(f"     Found {len(ultra_small_result.get('pain_points', []))} pain points")
                    logger.info(f"     Severity: {ultra_small_result.get('severity', 'none')}")

                    if ultra_small_result.get('pain_points'):
                        for pp in ultra_small_result['pain_points']:
                            recurring_terms = pp.get('recurring_terms', [])
                            question = pp.get('question', '')
                            count = pp.get('occurrence_count', 0)
                            logger.info(f"     Pain point: '{question}' (recurring terms: {recurring_terms}, count: {count})")

                except Exception as small_window_e:
                    logger.info(f"   ❌ Small window test failed: {small_window_e}")

                # Test with extremely permissive settings
                logger.info("\n   Testing with ultra-permissive settings...")
                try:
                    permissive_result = db_manager.detect_pain_points(
                        session_id=test_session_id,
                        threshold=0.1,  # Very low similarity required
                        min_occurrences=1,  # Only need 1 occurrence
                        time_window_days=0.00001  # Force temporal separation
                    )

                    logger.info(f"   Ultra-permissive result:")
                    logger.info(f"     Found {len(permissive_result.get('pain_points', []))} pain points")
                    logger.info(f"     Severity: {permissive_result.get('severity', 'none')}")
                    logger.info(f"     Total interactions analyzed: {permissive_result.get('total_interactions_analyzed', 0)}")

                    if permissive_result.get('pain_points'):
                        logger.info("   🎉 SUCCESS! Pain points detected with permissive settings:")
                        for i, pp in enumerate(permissive_result['pain_points']):
                            question = pp.get('question', '')
                            recurring_terms = pp.get('recurring_terms', [])
                            count = pp.get('occurrence_count', 0)
                            severity = pp.get('severity', 'unknown')
                            logger.info(f"     Pain point {i+1}: '{question}' (terms: {recurring_terms}, count: {count}, severity: {severity})")

                        for j, pp in enumerate(pain_points):
                            logger.info(f"       Pain point {j+1}:")
                            question_text = pp.get('question', 'NO QUESTION')
                            recurring_terms = pp.get('recurring_terms', [])
                            logger.info(f"         Question: '{question_text}'")
                            logger.info(f"         Recurring terms: {recurring_terms}")
                            logger.info(f"         Count: {pp.get('occurrence_count', 0)}")

                except Exception as permissive_e:
                    logger.info(f"   ❌ Permissive test failed: {permissive_e}")

                # Check what was actually stored in the database
                logger.info("\n📝 Step 3: Checking stored interactions for pain points")

                history = db_manager.get_conversation_history(test_session_id)
                logger.info(f"   Total interactions in history: {len(history)}")

                pain_points_found = False
                for i, interaction in enumerate(history):
                    logger.info(f"\n   Interaction {i+1}:")
                    logger.info(f"     Question: {interaction.get('question', '')[:50]}...")

                    # Process metadata
                    raw_metadata_from_db = interaction.get("metadata")
                    current_metadata: Dict[str, Any] = {}

                    if isinstance(raw_metadata_from_db, dict):
                        current_metadata = raw_metadata_from_db
                    elif isinstance(raw_metadata_from_db, str):
                        try:
                            current_metadata = json.loads(raw_metadata_from_db)
                        except:
                            pass

                    logger.info(f"     Metadata keys: {list(current_metadata.keys())}")

                    # Check for pain points
                    if 'pain_points' in current_metadata and current_metadata['pain_points']:
                        pain_points_found = True
                        pain_points = current_metadata['pain_points']
                        logger.info(f"     ✅ Pain points found: {len(pain_points)}")

                        for j, pp in enumerate(pain_points):
                            logger.info(f"       Pain point {j+1}:")
                            question_text = pp.get('question', 'NO QUESTION')
                            recurring_terms = pp.get('recurring_terms', [])
                            logger.info(f"         Question: '{question_text}'")
                            logger.info(f"         Recurring terms: {recurring_terms}")
                            logger.info(f"         Count: {pp.get('occurrence_count', 0)}")
                    else:
                        logger.info("     No pain points in metadata")

                if pain_points_found:
                    logger.info("\n✅ SUCCESS: Pain points found in interaction metadata!")
                else:
                    logger.info("\n⚠️ No pain points found in interaction metadata")
                    logger.info("   This suggests the temporal grouping is preventing detection")

                logger.info("\n✅ Pain point integration test completed!")

            else:
                logger.info(f"❌ Second interaction failed: {result2.get('error')}")
        else:
            logger.info(f"❌ First interaction failed: {result1.get('error')}")

        cleanup_memory()

    except Exception as e:
        logger.info(f"❌ Real database test failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n📝 Step 2.6: Deep algorithm debugging")

    # Get the pain detector and enable verbose logging
    pain_detector = db_manager.pain_point_detector

    # Test with manual debugging
    history = db_manager.get_conversation_history(test_session_id)
    logger.info(f"   History has {len(history)} interactions")

    # Check timestamps and temporal grouping
    for i, interaction in enumerate(history):
        timestamp = interaction.get('created_at', 'NO_TIME')
        question = interaction.get('question', '')[:60]
        logger.info(f"     {i+1}. {timestamp} - {question}...")

    # Test semantic chunking directly
    logger.info("\n   Testing semantic chunking:")
    q1 = "I feel inadequate at work every single day and it's affecting my confidence"
    q2 = "I'm still feeling inadequate every day and thinking about quitting my job"

    try:
        chunks1 = pain_detector._semantic_chunk_question(q1)
        chunks2 = pain_detector._semantic_chunk_question(q2)

        logger.info(f"   Q1 chunks: {chunks1}")
        logger.info(f"   Q2 chunks: {chunks2}")

        # Look for overlap
        overlap = set(chunks1) & set(chunks2)
        logger.info(f"   Overlapping chunks: {overlap}")

    except Exception as chunk_e:
        logger.info(f"   ❌ Chunking failed: {chunk_e}")

    # Test temporal grouping with exact timestamps
    try:
        temporal_groups = pain_detector._group_interactions_by_time(history, 30)
        logger.info(f"\n   Temporal groups: {len(temporal_groups)}")

        for i, group in enumerate(temporal_groups):
            interactions = group.get('interactions', [])
            logger.info(f"     Group {i}: {len(interactions)} interactions")
            for j, interaction in enumerate(interactions):
                question = interaction.get('question', '')[:40]
                logger.info(f"       {j+1}. {question}...")

    except Exception as temporal_e:
        logger.info(f"   ❌ Temporal grouping failed: {temporal_e}")

    # Test with forced minimal settings
    logger.info("\n   Testing with minimal requirements:")
    try:
        minimal_result = db_manager.detect_pain_points(
            session_id=test_session_id,
            threshold=0.01,  # Almost no similarity required
            min_occurrences=1,  # Just need 1 occurrence
            time_window_days=0.000001  # Tiny window
        )

        logger.info(f"   Minimal result: {minimal_result}")

        if minimal_result.get('pain_points'):
            logger.info("   🎉 SUCCESS with minimal settings!")
            for pp in minimal_result['pain_points']:
                question = pp.get('question', '')
                recurring_terms = pp.get('recurring_terms', [])
                logger.info(f"     - Question: '{question}' (terms: {recurring_terms})")
        else:
            logger.info("   ❌ Still no pain points with minimal settings")
            logger.info("   This indicates a fundamental algorithm issue")

    except Exception as minimal_e:
        logger.info(f"   ❌ Minimal test failed: {minimal_e}")
        import traceback
        traceback.print_exc()

    logger.info("\n📝 Step 2.7: Direct pain point detection test")

    # Test pain point detection directly with known good parameters
    direct_result = db_manager.detect_pain_points(
        session_id=test_session_id,
        threshold=0.5,  # Lower than 0.9491
        min_occurrences=2,
        time_window_days=1  # Small window to ensure different groups
    )

    logger.info(f"   Direct detection result: {direct_result}")
    logger.info(f"   Found {len(direct_result.get('pain_points', []))} pain points")
    logger.info(f"   Severity: {direct_result.get('severity', 'none')}")

    if direct_result.get('pain_points'):
        logger.info("   🎉 SUCCESS! Pain points detected with direct call")
        for i, pp in enumerate(direct_result['pain_points']):
            question = pp.get('question', '')
            recurring_terms = pp.get('recurring_terms', [])
            logger.info(f"     Pain point {i+1}: '{question}' (terms: {recurring_terms})")
    else:
        logger.info("   ❌ Still no pain points - checking temporal validation...")

    # Add this debugging code in your test
    logger.info("\n🔍 Debugging actual questions:")
    history = db_manager.get_conversation_history(test_session_id)
    for i, interaction in enumerate(history):
        question = interaction.get('question', '')
        logger.info(f"Question {i}: '{question}'")
        logger.info(f"Length: {len(question)} characters")
        logger.info("-" * 50)

    # Add this debugging after pain point detection
    logger.info("\n🔍 Checking if pain points were added to interaction metadata:")
    updated_history = db_manager.get_conversation_history(test_session_id)
    for i, interaction in enumerate(updated_history):
        metadata = interaction.get('metadata', {})
        logger.info(f"  Interaction {i} metadata: {metadata}")
        if 'pain_points' in metadata:
            logger.info(f"    ✅ Pain points found in metadata: {metadata['pain_points']}")
        else:
            logger.info(f"    ❌ No pain points in metadata")

if __name__ == "__main__":
    test_pain_point_integration_real()