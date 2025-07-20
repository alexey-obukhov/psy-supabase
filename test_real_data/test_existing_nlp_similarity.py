"""Test semantic similarity using existing NLP infrastructure."""

import os
import sys

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psy_supabase import get_package_logger

logger = get_package_logger(__name__)


def test_existing_nlp() -> None:
    """Test semantic similarity with existing spaCy setup."""

    logger.info("🔍 Testing existing spaCy semantic similarity")
    logger.info("=" * 50)

    # Test questions
    q1 = "I feel inadequate at work every single day"
    q2 = "I'm still feeling inadequate every day"

    logger.info(f"Question 1: {q1}")
    logger.info(f"Question 2: {q2}")

    try:
        from psy_supabase.utilities.nlp_utils import extract_entities, get_spacy_model

        # Get existing spaCy model
        nlp = get_spacy_model()

        if nlp:
            logger.info("✅ spaCy model loaded successfully")
            logger.info(f"   Model: {nlp.meta['name']} v{nlp.meta['version']}")

            # Test direct similarity
            doc1 = nlp(q1)
            doc2 = nlp(q2)
            similarity = doc1.similarity(doc2)

            logger.info(f"\n📝 Direct spaCy similarity: {similarity:.4f}")

            if similarity > 0.7:
                logger.info("✅ High similarity - should detect pain points!")
            elif similarity > 0.5:
                logger.info("⚠️ Medium similarity - might need threshold adjustment")
            else:
                logger.info("❌ Low similarity - may need different approach")

            # Test chunking with spaCy
            logger.info(f"\n📝 spaCy chunking analysis:")

            logger.info(f"\nQ1 noun chunks:")
            for chunk in doc1.noun_chunks:
                logger.info(f"   - '{chunk.text}'")

            logger.info(f"\nQ2 noun chunks:")
            for chunk in doc2.noun_chunks:
                logger.info(f"   - '{chunk.text}'")

            # Test entities
            entities1 = extract_entities(q1)
            entities2 = extract_entities(q2)

            logger.info(f"\nQ1 entities: {[e['text'] for e in entities1]}")
            logger.info(f"Q2 entities: {[e['text'] for e in entities2]}")

            # Test token-level similarity for key terms
            logger.info(f"\n📝 Key token similarities:")

            key_pairs = [
                ("feel", "feeling"),
                ("inadequate", "inadequate"),
                ("work", "day"),
                ("every", "every"),
                ("day", "day"),
            ]

            for w1, w2 in key_pairs:
                token1 = nlp(w1)[0]
                token2 = nlp(w2)[0]
                token_sim = token1.similarity(token2)
                logger.info(f"   '{w1}' vs '{w2}': {token_sim:.4f}")

        else:
            logger.info("❌ Failed to load spaCy model")

    except Exception as e:
        logger.info(f"❌ Error testing spaCy: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_existing_nlp()
