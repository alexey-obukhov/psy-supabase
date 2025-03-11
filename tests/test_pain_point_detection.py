import os
import sys
import uuid
from school_logging.log import ColoredLogger
from collections import Counter
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import torch


# Load environment variables
load_dotenv()

# Set up logging
logger = ColoredLogger("pain_point_detection_test")

device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Require Supabase credentials from environment
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.critical("SUPABASE_URL and SUPABASE_KEY must be set in environment or .env file")
    sys.exit(1)

from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.core.model_manager import ModelManager, EmbeddingProviderAdapter

# Test conversation scenarios with recurring themes
TEST_CONVERSATIONS = [
    # {
    #     "name": "Workplace Trauma Pattern",
    #     "questions": [
    #         "I had a really difficult day at work today. My boss criticized me in front of everyone again.",
    #         "Why do I always feel so nervous before team meetings? I'm prepared but still worry about being called out.",
    #         "Do you have any tips for handling workplace stress? I'm finding it hard to concentrate lately.",
    #         "My manager humiliated me in the meeting yesterday and I can't stop thinking about it. I feel like I'm walking on eggshells at my job."
    #     ],
    #     "expected_pain_point": {
    #         "themes": ["workplace", "criticism", "anxiety", "humiliation"],
    #         "approach_types": ["subtle", "gentle", "direct"]
    #     }
    # },
    {
        "name": "Relationship Confidence Pattern",
        "questions": [
            "How do I know if someone really likes me or is just being nice?",
            "I went on a date yesterday but I'm not sure if it went well. She haven't texted me back yet.",
            "My friends tell me I'm attractive but I don't feel confident when meeting new people. Any advice?",
            "I'm worried I'll be alone forever. Why do people never seem interested in me romantically?",
            "She texted me back and asked me out again! I'm excited but also nervous. What if I mess it up?",
            "Date went badly. She said she didn't feel a connection. I feel terrible."
        ],
        "expected_pain_point": {
            "themes": ["relationship", "confidence", "self-esteem", "rejection"],
            "approach_types": ["subtle", "gentle", "direct"]
        }
    },
    # {
    #     "name": "Grief Processing Pattern",
    #     "questions": [
    #         "Today would have been my mom's birthday. I miss her.",
    #         "Sometimes I think I hear my mom's voice even though she passed away last year. Is that normal?",
    #         "How long does grief usually last? I thought I was doing better but then found myself crying in the grocery store.",
    #         "I keep dreaming about my mom. In the dreams she's still alive and we're just doing normal things together. I wake up feeling terrible."
    #     ],
    #     "expected_pain_point": {
    #         "themes": ["grief", "loss", "mom", "dreams"],
    #         "approach_types": ["gentle", "direct"]
    #     }
    # },
    # {
    #     "name": "Self-Blame Pattern",
    #     "questions": [
    #         "How can I stop making so many mistakes at work?",
    #         "I feel like I'm disappointing everyone around me lately.",
    #         "What's wrong with me? I can never seem to get things right the first time.",
    #         "My friend says I'm too hard on myself but I just have high standards."
    #     ],
    #     "expected_pain_point": {
    #         "themes": ["mistake", "disappoint", "perfectionism", "self-criticism"],
    #         "approach_types": ["subtle", "gentle"]
    #     }
    # },
    # {
    #     "name": "Hidden Trauma Reference Pattern",
    #     "questions": [
    #         "I've been having trouble sleeping lately. Any suggestions?",
    #         "Sometimes I get really jumpy when I hear loud noises. Is that anxiety?",
    #         "I had a panic attack when a car backfired near me yesterday.",
    #         "Why do certain sounds make me feel like I'm in danger even when I'm safe?"
    #     ],
    #     "expected_pain_point": {
    #         "themes": ["anxiety", "trauma", "trigger", "panic"],
    #         "approach_types": ["subtle", "gentle", "direct"]
    #     }
    # }
]

class PainPointDetectionTester:
    """Tests the pain point detection capabilities of the RAG system."""
    
    def __init__(self):
        """Initialize test environment and components."""
        # Create a unique test user ID for this test run
        self.test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        logger.info(f"Using test user ID: {self.test_user_id}")
        
        # Create a unique schema name for testing based on the user ID
        self.test_session_id = f"test_pain_point_{uuid.uuid4().hex[:10]}"
        logger.info(f"Using test session ID: {self.test_session_id}")
        
        # Initialize components
        #self.model_manager = ModelManager()
        #self.embedding_provider = EmbeddingProviderAdapter()
        
        # Initialize database manager with Supabase credentials and test user
        self.db_manager = DatabaseManager(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            user_id=self.test_user_id
        )
        
        self.generator = TextGenerator(model_name = "microsoft/phi-1_5", device=device)

        # Initialize RAG processor with the test schema
        self.rag_processor = RAGProcessor(
            db_manager=self.db_manager,
            generator=self.generator,
            intelligent_processing_enabled=True
        )
        
        # Ensure we can access the database
        self.db_manager.schema_name = self.db_manager._sanitize_schema_name(self.test_session_id)
        
        # Set up test environment
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Set up the test environment, creating necessary tables."""
        logger.info("Setting up test environment...")
        
        # Initialize the database schema for testing
        self.db_manager.create_user_schema_sync()
        
        # Initialize knowledge base
        self.db_manager.initialize_knowledge_base(self.test_session_id)
        
        # Ensure vector indexes for proper pgvector functionality
        self.db_manager.ensure_vector_indexes(self.test_session_id)
        
        logger.info("Test environment setup complete")
    
    def _simulate_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a conversation and track pain point detection.
        
        Args:
            conversation: Dictionary with conversation name and questions
            
        Returns:
            Dictionary with conversation results
        """
        logger.info(f"Testing conversation: {conversation['name']}")
        
        results = {
            "name": conversation["name"],
            "exchanges": [],
            "pain_points_detected": 0,
            "templates_used": [],
            "first_detection_at": None
        }
        
        # Process each question in sequence
        for i, question in enumerate(conversation["questions"]):
            logger.info(f"Question {i+1}: {question[:50]}...")
            
            # Generate response through the RAG processor
            response = self.rag_processor.generate_response(
                user_question=question,
                device=device,
                question_id=i,
                session_id=self.test_session_id
            )
            
            # Get the most recent interaction's metadata
            try:
                history = self.db_manager.get_conversation_history(self.test_session_id)
                latest_interaction = history[-1] if history else {}
                
                # Convert metadata from string to dict if needed
                metadata = latest_interaction.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except Exception as e:
                        logger.error(f"Error parsing metadata JSON: {e}")
                        metadata = {}
                
                # Extract pain point information
                pain_point_detected = metadata.get("pain_point_detected", False)
                therapeutic_approach = metadata.get("therapeutic_approach", "none")
                template_used = metadata.get("template_used", "unknown")
                similarity = metadata.get("pain_point_similarity", 0)
                recurring_themes = metadata.get("recurring_themes", [])
                
                # Record results
                exchange_result = {
                    "question": question,
                    "response": response[:100] + "..." if response and len(response) > 100 else response if response else "No response",
                    "pain_point_detected": pain_point_detected,
                    "therapeutic_approach": therapeutic_approach,
                    "template_used": template_used,
                    "similarity": similarity,
                    "recurring_themes": recurring_themes
                }
                
                results["exchanges"].append(exchange_result)
                
                # Update summary statistics
                if pain_point_detected:
                    results["pain_points_detected"] += 1
                    if not results["first_detection_at"]:
                        results["first_detection_at"] = i + 1
                
                results["templates_used"].append(template_used)
                
                logger.info(f"Response generated. Pain point detected: {pain_point_detected}, " +
                           f"Approach: {therapeutic_approach}, Template: {template_used}")
                
            except Exception as e:
                logger.error(f"Error processing results: {e}")
                # Add more debug information
                if history:
                    logger.error(f"History type: {type(history)}")
                    if len(history) > 0:
                        logger.error(f"Last item type: {type(history[-1])}")
                        logger.error(f"Last item: {history[-1]}")
                
                # More robust handling for different response types
                try:
                    # Try to process even if we get strings instead of dictionaries
                    pain_point_detected = False
                    therapeutic_approach = "none"
                    template_used = "unknown"
                    similarity = 0
                    recurring_themes = []
                    
                    # Check if history exists and is a list
                    if history and isinstance(history, list) and len(history) > 0:
                        last_item = history[-1]
                        
                        # Handle string type
                        if isinstance(last_item, str):
                            logger.warning("Received string instead of dictionary in history")
                            # Try to parse if it looks like JSON
                            if last_item.startswith('{') and last_item.endswith('}'):
                                try:
                                    import json
                                    parsed_item = json.loads(last_item)
                                    if isinstance(parsed_item, dict):
                                        metadata = parsed_item.get("metadata", {})
                                        if isinstance(metadata, str):
                                            metadata = json.loads(metadata)
                                        pain_point_detected = metadata.get("pain_point_detected", False)
                                        therapeutic_approach = metadata.get("therapeutic_approach", "none")
                                        template_used = metadata.get("template_used", "unknown")
                                        similarity = metadata.get("pain_point_similarity", 0)
                                        recurring_themes = metadata.get("recurring_themes", [])
                                except:
                                    pass
                        # Handle dictionary type
                        elif isinstance(last_item, dict):
                            metadata = last_item.get("metadata", {})
                            # Handle metadata as string
                            if isinstance(metadata, str):
                                try:
                                    import json
                                    metadata = json.loads(metadata)
                                except:
                                    metadata = {}
                            
                            pain_point_detected = metadata.get("pain_point_detected", False) if isinstance(metadata, dict) else False
                            therapeutic_approach = metadata.get("therapeutic_approach", "none") if isinstance(metadata, dict) else "none"
                            template_used = metadata.get("template_used", "unknown") if isinstance(metadata, dict) else "unknown"
                            similarity = metadata.get("pain_point_similarity", 0) if isinstance(metadata, dict) else 0
                            recurring_themes = metadata.get("recurring_themes", []) if isinstance(metadata, dict) else []
                    
                    # Record results with the data we could extract
                    exchange_result = {
                        "question": question,
                        "response": response[:100] + "..." if len(response) > 100 else response,
                        "pain_point_detected": pain_point_detected,
                        "therapeutic_approach": therapeutic_approach,
                        "template_used": template_used,
                        "similarity": similarity,
                        "recurring_themes": recurring_themes
                    }
                    
                    results["exchanges"].append(exchange_result)
                    
                    # Update summary statistics
                    if pain_point_detected:
                        results["pain_points_detected"] += 1
                        if not results["first_detection_at"]:
                            results["first_detection_at"] = i + 1
                    
                    results["templates_used"].append(template_used)
                    
                except Exception as e2:
                    logger.error(f"Second attempt at processing results failed: {e2}")
                    # Simple fallback with just the question and response
                    results["exchanges"].append({
                        "question": question,
                        "response": response[:100] + "..." if response and len(response) > 100 else response if response else "No response",
                        "error": str(e)
                    })
        
        return results
    
    def run_tests(self) -> List[Dict[str, Any]]:
        """
        Run all test conversations and collect results.
        
        Returns:
            List of result dictionaries for each conversation
        """
        all_results = []
        
        for conversation in TEST_CONVERSATIONS:
            # Add a small delay between conversations
            logger.info(f"Starting test for: {conversation['name']}")
            
            # Simulate the conversation
            result = self._simulate_conversation(conversation)
            all_results.append(result)
            
            # Log summary of this conversation test
            logger.info(f"Completed test for: {conversation['name']}")
            logger.info(f"  Pain points detected: {result['pain_points_detected']} out of {len(conversation['questions'])}")
            if result["first_detection_at"]:
                logger.info(f"  First detected at question #{result['first_detection_at']}")
            logger.info(f"  Templates used: {', '.join(result['templates_used'])}")
            logger.info("----------------------------------------")
        
        return all_results
    
    def cleanup(self):
        """Clean up test environment to avoid cluttering the database."""
        logger.info("Cleaning up test environment...")
        
        try:
            # Drop the test schema
            self.db_manager.drop_schema(self.test_session_id)
            logger.info(f"Dropped test schema: {self.db_manager.schema_name}")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")

def analyze_test_results(results: List[Dict[str, Any]]):
    """
    Analyze and print summary of test results.
    
    Args:
        results: List of test results from run_tests()
    """
    logger.info("=== PAIN POINT DETECTION TEST RESULTS ===")
    logger.info(f"Conversations tested: {len(results)}")
    
    # Overall statistics
    total_exchanges = sum(len(r["exchanges"]) for r in results)
    total_detected = sum(r["pain_points_detected"] for r in results)
    detection_rate = (total_detected / total_exchanges) * 100 if total_exchanges > 0 else 0
    
    logger.info(f"Total exchanges: {total_exchanges}")
    logger.info(f"Total pain points detected: {total_detected}")
    logger.info(f"Overall detection rate: {detection_rate:.2f}%")
    
    # Analyze each conversation
    logger.info("\nDetailed results by conversation:")
    for result in results:
        logger.info(f"\n{result['name']}:")
        logger.info(f"  Detection rate: {(result['pain_points_detected'] / len(result['exchanges'])) * 100:.2f}%")
        logger.info(f"  First detected at: Question #{result['first_detection_at'] if result['first_detection_at'] else 'N/A'}")
        
        # Analyze themes detected
        all_themes = []
        for exchange in result["exchanges"]:
            if exchange.get("recurring_themes"):
                all_themes.extend(exchange["recurring_themes"])
        
        if all_themes:
            theme_counts = Counter(all_themes)
            logger.info(f"  Top detected themes: {', '.join([f'{t}({c})' for t, c in theme_counts.most_common(3)])}")
        
        # Check templates used
        template_counts = Counter(result["templates_used"])
        logger.info(f"  Templates used: {', '.join([f'{t}({c})' for t, c in template_counts.most_common()])}")

def main():
    """Run the pain point detection tests."""
    logger.info("Starting pain point detection tests")

    # Verify Supabase credentials before proceeding
    if not supabase_url or not supabase_key:
        logger.critical("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        return

    try:
        tester = PainPointDetectionTester()
        results = tester.run_tests()
        analyze_test_results(results)
        tester.cleanup()
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()