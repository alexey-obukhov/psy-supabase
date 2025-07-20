"""Debug the semantic chunking method specifically."""

import datetime
import os
import sys
import uuid
from typing import cast

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from psy_supabase import get_package_logger
from psy_supabase.core.database import DatabaseManager

load_dotenv()

logger = get_package_logger(__name__)


def debug_chunking() -> None:
    """Debug what semantic chunking is actually doing."""

    logger.info("🔍 Debugging semantic chunking method")
    logger.info("=" * 50)

    # Setup minimal database
    supabase_url = cast(str, os.environ.get("SUPABASE_URL"))
    supabase_key = cast(str, os.environ.get("SUPABASE_KEY"))
    test_user_id = f"debug_user_{uuid.uuid4().hex[:8]}"

    db_manager = DatabaseManager(supabase_url=supabase_url, supabase_key=supabase_key, user_id=test_user_id)

    pain_detector = db_manager.pain_point_detector
    db_manager.schema_name = db_manager.schema_name
    db_manager.create_user_schema_sync()

    # Test questions
    q1 = "I feel inadequate at work every single day and it's affecting my confidence"
    q2 = "I'm still feeling inadequate every day and thinking about quitting my job"

    logger.info(f"Question 1: {q1}")
    logger.info(f"Question 2: {q2}")

    # Insert both questions as interactions and get their IDs
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    interaction_ids = []
    questions = [q1, q2]
    for idx, question in enumerate(questions):
        data_point = {
            "user_id": test_user_id,
            "question": question,
            "context": "",
            "answer": "",
            "created_at": now,
            "metadata": {},
        }
        db_manager.add_interaction_rpc(data_point)
        logger.info(f"Inserted interaction for question {idx}: {question}")
        interaction_ids.append(idx)  # or just use idx directly

    # Use the second interaction (index 1) as the current one for updating
    current_interaction_index = 1
    # Now you can use current_interaction_index to refer to the second question/interaction

    # Test semantic chunking directly
    try:
        logger.info("\n📝 Testing _semantic_chunk_question method directly...")

        chunks1 = pain_detector._semantic_chunk_question(q1)
        chunks2 = pain_detector._semantic_chunk_question(q2)

        logger.info(f"\nQ1 chunks ({len(chunks1)}):")
        for i, chunk in enumerate(chunks1):
            logger.info(f"  {i+1}. '{chunk}'")

        logger.info(f"\nQ2 chunks ({len(chunks2)}):")
        for i, chunk in enumerate(chunks2):
            logger.info(f"  {i+1}. '{chunk}'")

        logger.info(f"\n✅ Chunking is working - {len(chunks1)} + {len(chunks2)} chunks created")

        # Test the actual comparison logic using existing methods
        logger.info("\n📝 Testing chunk comparison logic...")

        set1 = set(chunk.strip().lower() for chunk in chunks1 if len(chunk.strip()) > 1)
        set2 = set(chunk.strip().lower() for chunk in chunks2 if len(chunk.strip()) > 1)

        intersection = set1 & set2
        union = set1 | set2

        logger.info(f"\nSet 1 ({len(set1)}): {sorted(list(set1))}")
        logger.info(f"Set 2 ({len(set2)}): {sorted(list(set2))}")
        logger.info(f"Intersection ({len(intersection)}): {sorted(list(intersection))}")
        logger.info(f"Union ({len(union)}): {sorted(list(union))}")

        if intersection:
            similarity = len(intersection) / len(union)
            logger.info(f"\n✅ Jaccard Similarity: {similarity:.3f}")
            logger.info(f"   Matching chunks: {list(intersection)}")

            if similarity > 0.3:  # 30% threshold
                logger.info("🎯 PAIN POINT DETECTED! High chunk overlap")
                painpoint_chunks = [chunk for chunk in intersection if len(chunk) > 1]
                logger.info(f"Writing painpoint_chunks to metadata for interaction: {current_interaction_index}")

                # Prepare your metadata with painpoint_chunks
                metadata = {
                    "painpoint_chunks": painpoint_chunks,
                    # ... any other metadata fields ...
                }

                # Use the update_interaction_metadata RPC
                db_manager.supabase.rpc(
                    "update_interaction_metadata",
                    {
                        "p_schema_name": db_manager.schema_name,
                        "p_interaction_id": current_interaction_index,
                        "p_metadata": metadata,
                    },
                ).execute()
                logger.info(f"✅ Updated metadata with painpoint_chunks for interaction id {current_interaction_index}")
            else:
                logger.info("📝 Low similarity - not a pain point")
        else:
            logger.info("\n❌ No matching chunks found")

        # Test recurring terms extraction using existing method
        logger.info("\n📝 Testing recurring terms extraction...")
        all_chunks = chunks1 + chunks2

        try:
            recurring_terms = pain_detector._extract_repeating_terms(all_chunks)
            logger.info(f"All chunks combined: {len(all_chunks)}")
            logger.info(f"Extracted recurring terms: {recurring_terms}")

            if recurring_terms:
                logger.info("✅ Successfully extracted recurring terms!")
            else:
                logger.info("❌ No recurring terms found - check _extract_repeating_terms method")

        except Exception as e:
            logger.info(f"❌ Error in recurring terms extraction: {e}")

        # Test the similarity calculation using existing method
        logger.info("\n📝 Testing similarity calculation...")
        try:
            similarity_score = pain_detector._calculate_max_chunk_similarity(chunks1, chunks2)
            logger.info(f"Similarity score from _calculate_max_chunk_similarity: {similarity_score:.3f}")

            if similarity_score > 0.5:
                logger.info("🎯 Pain point detected by similarity method!")
            else:
                logger.info("📝 Below pain point threshold")

        except Exception as e:
            logger.info(f"❌ Error in similarity calculation: {e}")

        # Test theme extraction using existing mappings
        logger.info("\n📝 Testing theme extraction...")
        try:
            # Use existing theme keywords
            theme_keywords = pain_detector.theme_keywords
            logger.info(f"Available themes: {list(theme_keywords.keys())}")

            # Test theme detection for these chunks
            combined_text = q1 + " " + q2
            detected_themes = []

            for theme, keywords in theme_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in combined_text.lower():
                        detected_themes.append(theme)
                        break

            logger.info(f"Detected themes: {detected_themes}")

        except Exception as e:
            logger.info(f"❌ Error in theme extraction: {e}")

    except Exception as e:
        logger.info(f"❌ Chunking test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_chunking()
