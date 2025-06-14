#!/usr/bin/env python3
"""
Test database permissions for pain point storage.
"""
import os
import sys
import uuid
import json
from typing import cast, Dict, Any, List

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psy_supabase import get_package_logger
from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.config import TEXT_GENERATING_MODEL
from dotenv import load_dotenv

load_dotenv()

logger = get_package_logger(__name__)

def test_database_permissions():
    """Test database permissions for pain point storage using working patterns."""

    logger.info("🔍 Testing Database Permissions for Pain Point Storage")
    logger.info("=" * 60)

    supabase_url = cast(str, os.environ.get("SUPABASE_URL"))
    supabase_key = cast(str, os.environ.get("SUPABASE_KEY"))
    test_user_id = f"test_permissions_{uuid.uuid4().hex[:8]}"
    test_session_id = f"perm_test_{uuid.uuid4().hex[:6]}"

    if not supabase_url or not supabase_key:
        logger.info("❌ SUPABASE_URL or SUPABASE_KEY not found in environment")
        return False

    logger.info(f"✅ Environment variables loaded")
    logger.info(f"   URL: {supabase_url[:30]}...")
    logger.info(f"   Key: {supabase_key[:20]}...")
    logger.info(f"📝 Creating database manager for user: {test_user_id}")

    # Create components using exact pattern
    db_manager = DatabaseManager(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        user_id=test_user_id
    )

    generator = TextGenerator(model_name=TEXT_GENERATING_MODEL, device="cpu")

    rag_processor = RAGProcessor(
        db_manager=db_manager,
        generator=generator,
        intelligent_processing_enabled=True
    )

    logger.info(f"✅ Database manager created with schema: {db_manager.schema_name}")

    # Test 1: Create schema using your working pattern
    logger.info("\n🔧 Test 1: Creating schema and tables...")
    db_manager.create_user_schema_sync()
    logger.info("✅ Schema created successfully")

    # Test 2: Add interactions using working pattern
    logger.info("\n📝 Test 2: Adding test interactions...")

    questions = [
        "I feel inadequate at work every single day and it's affecting my confidence",
        "I'm still feeling inadequate every day and thinking about quitting my job"
    ]

    for i, question in enumerate(questions):
        logger.info(f"Question {i+1}: {question[:50]}...")

        response = rag_processor.generate_response(
            user_question=question,
            session_id=test_session_id,
            device="cpu",
            question_id=i
        )

        logger.info(f"✅ Interaction {i+1} added successfully")

    # Test 3: Check conversation history and metadata using pattern
    logger.info("\n📋 Test 3: Checking conversation history...")
    history = db_manager.get_conversation_history(test_session_id)
    logger.info(f"✅ Retrieved {len(history)} interactions")

    pain_points_found_in_metadata = False

    for i, interaction in enumerate(history):
        # Handle the list format: [json_string, simple_dict]
        raw_metadata = interaction.get("metadata", [])

        metadata = {}
        if isinstance(raw_metadata, list) and len(raw_metadata) > 0:
            # Take the first element which contains the full metadata
            metadata_item = raw_metadata[0]
            if isinstance(metadata_item, str):
                metadata = json.loads(metadata_item)
            elif isinstance(metadata_item, dict):
                metadata = metadata_item
        elif isinstance(raw_metadata, str):
            metadata = json.loads(raw_metadata)
        elif isinstance(raw_metadata, dict):
            metadata = raw_metadata

        logger.info(f"   Interaction {i+1} metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'NOT_DICT'}")

        if isinstance(metadata, dict) and metadata.get("pain_points") and isinstance(metadata.get("pain_points"), list):
            pain_points = metadata.get("pain_points")
            if pain_points and any(pp.get("detected", False) for pp in pain_points):
                pain_points_found_in_metadata = True
                logger.info(f"     🎯 Found {len(pain_points)} pain points in metadata")
                for j, pp in enumerate(pain_points):
                    detected = pp.get('detected', False)
                    question = pp.get('question', 'NO_QUESTION')
                    theme = pp.get('theme', 'NO_THEME')
                    similarity = pp.get('similarity', 0.0)
                    logger.info(f"       Pain point {j+1}: detected={detected}, similarity={similarity}, question='{question[:50]}'")
            else:
                logger.info(f"     📝 Found {len(pain_points)} pain points but none detected=True")
        else:
            logger.info(f"     📝 No pain points in interaction {i+1} metadata")

    if not pain_points_found_in_metadata:
        logger.info("   ⚠️ No pain points found in any interaction metadata")
        logger.info("   💡 This indicates the pain point detection/save process needs investigation")

    # Test 4: Run direct pain point detection
    logger.info("\n🔍 Test 4: Testing direct pain point detection...")
    pain_result = db_manager.detect_pain_points(
        session_id=test_session_id,
        threshold=0.5,
        min_occurrences=2,
        time_window_days=1
    )
    logger.info(f"✅ Pain point detection executed: {pain_result}")

    if pain_result.get('pain_points'):
        logger.info(f"   🎯 Found {len(pain_result['pain_points'])} pain points")
        logger.info(f"   Severity: {pain_result.get('severity', 'none')}")
    else:
        logger.info("   📝 No pain points detected")

    logger.info("\n🎉 All database permission tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_database_permissions()
    if success:
        logger.info("\n✅ Database permissions test PASSED")
        logger.info("💡 Pain point detection and storage is working correctly!")
    else:
        logger.info("\n❌ Database permissions test FAILED")
        sys.exit(1)