"""Simple diagnostic test for pain point similarity."""

import os
import sys
import uuid
from typing import cast

import numpy as np

# Configure logging
from psy_supabase import get_package_logger

logger = get_package_logger(__name__)
# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.model_manager import get_embedding_provider

load_dotenv()


def test_similarity_only() -> None:
    """Test just the similarity calculation."""

    logger.info("🔍 Testing pain point similarity calculation only")
    logger.info("=" * 50)

    # Setup minimal database
    supabase_url = cast(str, os.environ.get("SUPABASE_URL"))
    supabase_key = cast(str, os.environ.get("SUPABASE_KEY"))
    test_user_id = f"debug_user_{uuid.uuid4().hex[:8]}"

    db_manager = DatabaseManager(supabase_url=supabase_url, supabase_key=supabase_key, user_id=test_user_id)

    # Get the pain detector
    pain_detector = db_manager.pain_point_detector

    # Test questions
    q1 = "I feel inadequate at work every single day and it's affecting my confidence"
    q2 = "I'm still feeling inadequate every day and thinking about quitting my job"

    logger.info(f"Question 1: {q1}")
    logger.info(f"Question 2: {q2}")

    # Test semantic chunking
    try:
        logger.info("\n📝 Testing semantic chunking...")
        chunks1 = pain_detector._semantic_chunk_question(q1)
        chunks2 = pain_detector._semantic_chunk_question(q2)

        logger.info(f"Q1 chunks ({len(chunks1)}): {chunks1}")
        logger.info(f"Q2 chunks ({len(chunks2)}): {chunks2}")

        # Check for obvious overlaps
        max_chunk_similarity = 0
        for c1 in chunks1:
            for c2 in chunks2:
                if len(c1.strip()) > 5 and len(c2.strip()) > 5:
                    # Calculate embedding similarity
                    emb1 = get_embedding_provider().generate_embedding(c1)
                    emb2 = get_embedding_provider().generate_embedding(c2)

                    emb1_norm = np.array(emb1) / np.linalg.norm(emb1)
                    emb2_norm = np.array(emb2) / np.linalg.norm(emb2)
                    chunk_sim = np.dot(emb1_norm, emb2_norm)

                    max_chunk_similarity = max(max_chunk_similarity, chunk_sim)

        if max_chunk_similarity < 0.7:
            logger.info("❌ NO OVERLAPPING CHUNKS")
        else:
            logger.info("✅ Found overlapping chunks")

    except Exception as e:
        logger.info("❌ Chunking failed: %s", e)
        return

    # Test embedding similarity directly
    try:
        logger.info("\n📝 Testing embedding similarity...")

        # Get embedding provider
        embed_provider = get_embedding_provider()

        # Generate embeddings for the full questions
        emb1 = embed_provider.generate_embedding(q1)
        emb2 = embed_provider.generate_embedding(q2)

        # Normalize embeddings
        emb1_norm = np.array(emb1) / np.linalg.norm(emb1)
        emb2_norm = np.array(emb2) / np.linalg.norm(emb2)

        # Calculate similarity
        similarity = np.dot(emb1_norm, emb2_norm)

        logger.info(f"Direct embedding similarity: {similarity:.4f}")

        if similarity > 0.7:
            logger.info("✅ High similarity - algorithm should detect this")
        elif similarity > 0.5:
            logger.info("⚠️ Medium similarity - might need lower threshold")
        else:
            logger.info("❌ Low similarity - explains why detection fails")

    except Exception as e:
        logger.info(f"❌ Embedding similarity failed: {e}")

    # Test chunk-level similarity
    try:
        logger.info("\n📝 Testing chunk-level similarity...")

        if chunks1 and chunks2:
            # Test similarity between overlapping chunks
            embed_provider = get_embedding_provider()

            max_similarity = 0
            best_pair = None

            for c1 in chunks1:
                for c2 in chunks2:
                    if len(c1.strip()) > 5 and len(c2.strip()) > 5:
                        emb_c1 = embed_provider.generate_embedding(c1)
                        emb_c2 = embed_provider.generate_embedding(c2)

                        # Calculate similarity
                        emb_c1_norm = np.array(emb_c1) / np.linalg.norm(emb_c1)
                        emb_c2_norm = np.array(emb_c2) / np.linalg.norm(emb_c2)
                        chunk_sim = np.dot(emb_c1_norm, emb_c2_norm)

                        if chunk_sim > max_similarity:
                            max_similarity = chunk_sim
                            best_pair = (c1, c2)

                        logger.info(f"   '{c1}' vs '{c2}': {chunk_sim:.4f}")

            logger.info(f"\nBest chunk similarity: {max_similarity:.4f}")
            if best_pair:
                logger.info(f"Best pair: '{best_pair[0]}' vs '{best_pair[1]}'")

            if max_similarity > 0.7:
                logger.info("✅ High chunk similarity found")
            else:
                logger.info("❌ All chunk similarities below 0.7 threshold")

    except Exception as e:
        logger.info(f"❌ Chunk similarity failed: {e}")


if __name__ == "__main__":
    test_similarity_only()
