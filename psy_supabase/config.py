"""Configuration file for the psy_supabase program.

This module contains all configurable parameters for the application.
Modify these settings to change application behavior without code changes.

Usage Examples:
--------------

1. Change pain point detection sensitivity:
   ```python
   # More sensitive detection (finds more pain points)
   PAIN_POINT_DETECTION["similarity_threshold"] = 0.5
   PAIN_POINT_DETECTION["min_occurrences"] = 1

   # Less sensitive detection (finds only obvious pain points)
   PAIN_POINT_DETECTION["similarity_threshold"] = 0.8
   PAIN_POINT_DETECTION["min_occurrences"] = 3
   ```

2. Optimize for performance vs accuracy:
   ```python
   # High performance (faster, less accurate)
   PAIN_POINT_DETECTION["chunking"]["include_full_question"] = False
   PAIN_POINT_DETECTION["chunking"]["max_chunks_per_question"] = 5
   PAIN_POINT_DETECTION["chunking"]["early_exit_on_perfect_match"] = True

   # High accuracy (slower, more accurate)
   PAIN_POINT_DETECTION["chunking"]["include_full_question"] = True
   PAIN_POINT_DETECTION["chunking"]["max_chunks_per_question"] = 12
   PAIN_POINT_DETECTION["chunking"]["early_exit_on_perfect_match"] = False
   ```

3. Customize response generation:
   ```python
   # Longer, more detailed responses
   RESPONSE_CONFIG["max_response_length"] = 800
   RESPONSE_CONFIG["temperature"] = 0.9  # More creative

   # Shorter, more focused responses
   RESPONSE_CONFIG["max_response_length"] = 200
   RESPONSE_CONFIG["temperature"] = 0.5  # More conservative
   ```

4. Adjust toxicity detection:
   ```python
   # Stricter toxicity filtering
   TOXICITY_CONFIG["threshold"] = 0.6

   # More lenient toxicity filtering
   TOXICITY_CONFIG["threshold"] = 0.9
   ```
"""

# Default response generation settings
DEFAULT_TOPIC = "supportive_listening"
DEFAULT_EMOTION = "concern"
DEFAULT_APPROACH = "empathy_validation"
DEFAULT_THEME = "general_support"

# Model configurations
TEXT_GENERATING_MODEL = "rasyosef/Phi-1_5-Instruct-v0.1"  # Beware templates should be adjusted regarding this model
TOXIC_CLASSIFICATION_MODEL = "facebook/roberta-hate-speech-dynabench-r4-target"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Pain Point Detection Configuration
# =================================
# Controls how the system detects recurring emotional patterns using semantic clustering.
PAIN_POINT_DETECTION = {
    # Core semantic clustering parameters
    "similarity_threshold": 0.6,  # 0.0-1.0: Minimum similarity to group items into same cluster
    "min_occurrences": 2,  # Minimum cluster size to be considered a pain point
    "time_window_days": 30,  # Temporal window for clustering analysis
    # Semantic chunking controls
    # Controls how user questions are semantically decomposed for clustering
    "chunking": {
        # SEMANTIC ANALYSIS TRADE-OFFS:
        "include_full_question": True,  # Include full context in semantic analysis
        "min_chunk_length": 4,  # Ignore semantically insignificant short chunks
        "max_chunks_per_question": 8,  # Limit semantic chunks (affects clustering granularity)
        "remove_redundant_chunks": True,  # Remove semantically duplicate chunks
        # Semantic feature extraction
        "extract_noun_phrases": True,  # Extract entity-based semantic chunks
        "extract_emotional_phrases": True,  # Extract emotion-based semantic chunks
        "extract_temporal_phrases": True,  # Extract time-based semantic chunks
        "extract_action_phrases": True,  # Extract action-based semantic chunks
        # Clustering optimization
        "early_exit_on_perfect_match": True,  # Stop semantic comparison at perfect similarity
        "log_only_best_matches": True,  # Log only highest semantic similarities
    },
}

# Database Configuration
# ======================
# Controls database operations and connection handling
DATABASE_CONFIG = {
    "interaction_batch_size": 100,  # Process interactions in batches of this size
    "max_sessions_per_user": 50,  # Limit sessions per user to prevent abuse
    "session_timeout_days": 7,  # Auto-expire sessions after this many days
}

# Response Generation Configuration
# ================================
# Controls AI response generation behavior
RESPONSE_CONFIG = {
    "max_response_length": 500,  # Maximum characters in generated response
    "temperature": 0.7,  # 0.0-1.0: Lower = more conservative, higher = more creative
    "max_context_tokens": 2048,  # Maximum context length for AI model
}

# Toxicity Detection Configuration
# ===============================
# Controls content filtering and safety measures
TOXICITY_CONFIG = {
    "threshold": 0.8,  # 0.0-1.0: Lower = stricter filtering, higher = more permissive
    "check_user_input": True,  # Scan user messages for toxicity
    "check_generated_response": True,  # Scan AI responses for toxicity
}

# RAG Processing Configuration
# ===========================
# Controls retrieval-augmented generation behavior and caching
RAG_CONFIG = {
    # Vector retrieval settings
    "similarity_threshold": 0.7,  # Minimum similarity for relevant documents
    "max_knowledge_chars": 500,  # Maximum characters for knowledge context
    "max_conversation_exchanges": 2,  # Maximum conversation exchanges to include
    # Performance optimization
    "vector_cache_enabled": True,  # Enable vector caching for similar questions
    "similarity_cache_timeout": 60,  # Cache timeout in seconds
    "deduplication_window": 5,  # Seconds to prevent duplicate searches
    "max_cache_entries": 10,  # Maximum cache entries to keep
    # Context enhancement
    "enable_conversation_context": True,  # Include conversation history
    "enable_knowledge_context": True,  # Include vector similarity search
    "enable_hot_topics": True,  # Include hot topic detection
    "enable_associative_memory": True,  # Include associative memory retrieval
    # Logging and debugging
    "log_similarity_searches": True,  # Log similarity search operations
    "log_cache_hits": True,  # Log cache hit/miss information
    "log_context_sizes": True,  # Log context size information
}

# CONFIGURATION IMPACT GUIDE
# ==========================
#
# FOR DEVELOPMENT/TESTING:
# - Set similarity_threshold = 0.5, min_occurrences = 1 (more sensitive)
# - Set log_only_best_matches = False (full debug logging)
# - Set include_full_question = True (maximum accuracy)
# - Set RAG_CONFIG["log_similarity_searches"] = True (debug RAG operations)
# - Set RAG_CONFIG["log_cache_hits"] = True (track cache performance)
# - Set RAG_CONFIG["deduplication_window"] = 0 (disable deduplication for testing)
#
# FOR PRODUCTION (HIGH PERFORMANCE):
# - Set include_full_question = False (50% performance boost)
# - Set max_chunks_per_question = 5 (faster processing)
# - Set early_exit_on_perfect_match = True (stop early on matches)
# - Set log_only_best_matches = True (cleaner logs)
# - Set RAG_CONFIG["vector_cache_enabled"] = True (enable caching)
# - Set RAG_CONFIG["deduplication_window"] = 5 (prevent duplicate searches)
# - Set RAG_CONFIG["max_cache_entries"] = 20 (larger cache for better performance)
# - Set RAG_CONFIG["log_similarity_searches"] = False (reduce log noise)
#
# FOR PRODUCTION (HIGH ACCURACY):
# - Set similarity_threshold = 0.7 (more precise matches)
# - Set min_occurrences = 3 (require more evidence)
# - Set include_full_question = True (full context)
# - Set max_chunks_per_question = 12 (more thorough analysis)
# - Set RAG_CONFIG["similarity_threshold"] = 0.8 (higher quality context)
# - Set RAG_CONFIG["max_knowledge_chars"] = 1000 (more context)
# - Set RAG_CONFIG["max_conversation_exchanges"] = 5 (richer history)
#
# MEMORY/RESOURCE OPTIMIZATION:
# - Set interaction_batch_size = 50 (smaller batches)
# - Set max_response_length = 300 (shorter responses)
# - Set max_context_tokens = 1024 (less context)
# - Set RAG_CONFIG["max_knowledge_chars"] = 300 (limit context size)
# - Set RAG_CONFIG["max_cache_entries"] = 5 (small cache)
# - Set RAG_CONFIG["similarity_cache_timeout"] = 30 (quick cache expiry)
# - Set RAG_CONFIG["enable_associative_memory"] = False (disable heavy features)
#
# SENSITIVITY ADJUSTMENT:
# - Depression/Anxiety detection: similarity_threshold = 0.6, min_occurrences = 2
# - Stress pattern detection: similarity_threshold = 0.5, min_occurrences = 3
# - Crisis intervention: similarity_threshold = 0.7, min_occurrences = 1
#
# SEMANTIC CLUSTERING IMPACT GUIDE
# ================================
#
# FOR SENSITIVE CLUSTERING (finds more pain point clusters):
# - Set similarity_threshold = 0.5 (lower threshold = more clusters)
# - Set min_occurrences = 1 (smaller cluster sizes accepted)
# - Set include_full_question = True (full semantic context)
#
# FOR PRECISE CLUSTERING (finds only strong pain point clusters):
# - Set similarity_threshold = 0.8 (higher threshold = fewer, stronger clusters)
# - Set min_occurrences = 3 (larger cluster sizes required)
# - Set max_chunks_per_question = 12 (more semantic features for precision)
#
# FOR PERFORMANCE-OPTIMIZED CLUSTERING:
# - Set include_full_question = False (faster semantic analysis)
# - Set max_chunks_per_question = 5 (fewer semantic comparisons)
# - Set early_exit_on_perfect_match = True (stop at perfect semantic match)
#
# SEMANTIC GRANULARITY CONTROL:
# - High granularity: max_chunks_per_question = 12 (fine-grained clusters)
# - Low granularity: max_chunks_per_question = 5 (broad clusters)
# - Context-aware: include_full_question = True (holistic clustering)
# - Feature-based: include_full_question = False (chunk-based clustering)
#
# CLUSTERING SENSITIVITY BY USE CASE:
# - Depression patterns: similarity_threshold = 0.6, min_occurrences = 2
# - Anxiety patterns: similarity_threshold = 0.65, min_occurrences = 2
# - Stress patterns: similarity_threshold = 0.5, min_occurrences = 3
# - Crisis intervention: similarity_threshold = 0.7, min_occurrences = 1
#
# RAG PROCESSING OPTIMIZATION GUIDE
# =================================
#
# FOR HIGH-QUALITY CONTEXT RETRIEVAL:
# - Set RAG_CONFIG["similarity_threshold"] = 0.8 (only highly relevant documents)
# - Set RAG_CONFIG["max_knowledge_chars"] = 1000 (rich context)
# - Set RAG_CONFIG["max_conversation_exchanges"] = 5 (full conversation history)
# - Set RAG_CONFIG["enable_conversation_context"] = True (include chat history)
# - Set RAG_CONFIG["enable_knowledge_context"] = True (include vector search)
# - Set RAG_CONFIG["enable_associative_memory"] = True (include memory associations)
#
# FOR FAST RESPONSE GENERATION:
# - Set RAG_CONFIG["similarity_threshold"] = 0.6 (accept more documents faster)
# - Set RAG_CONFIG["max_knowledge_chars"] = 300 (shorter context)
# - Set RAG_CONFIG["max_conversation_exchanges"] = 1 (minimal history)
# - Set RAG_CONFIG["vector_cache_enabled"] = True (cache frequently used)
# - Set RAG_CONFIG["deduplication_window"] = 10 (prevent redundant searches)
# - Set RAG_CONFIG["enable_hot_topics"] = False (disable complex processing)
#
# FOR MEMORY-CONSTRAINED ENVIRONMENTS:
# - Set RAG_CONFIG["max_cache_entries"] = 3 (small cache)
# - Set RAG_CONFIG["similarity_cache_timeout"] = 30 (quick expiry)
# - Set RAG_CONFIG["max_knowledge_chars"] = 200 (minimal context)
# - Set RAG_CONFIG["enable_associative_memory"] = False (reduce memory usage)
# - Set RAG_CONFIG["log_similarity_searches"] = False (reduce logging overhead)
# - Set RAG_CONFIG["log_context_sizes"] = False (reduce processing)
#
# FOR DEBUGGING RAG ISSUES:
# - Set RAG_CONFIG["log_similarity_searches"] = True (track all searches)
# - Set RAG_CONFIG["log_cache_hits"] = True (monitor cache performance)
# - Set RAG_CONFIG["log_context_sizes"] = True (verify context building)
# - Set RAG_CONFIG["deduplication_window"] = 0 (disable for testing)
# - Set RAG_CONFIG["vector_cache_enabled"] = False (force fresh searches)
#
# RAG CACHING STRATEGIES:
# ----------------------
#
# Aggressive Caching (for stable content):
# - Set RAG_CONFIG["vector_cache_enabled"] = True
# - Set RAG_CONFIG["similarity_cache_timeout"] = 300 (5 minutes)
# - Set RAG_CONFIG["max_cache_entries"] = 50
# - Set RAG_CONFIG["deduplication_window"] = 15
#
# Conservative Caching (for dynamic content):
# - Set RAG_CONFIG["vector_cache_enabled"] = True
# - Set RAG_CONFIG["similarity_cache_timeout"] = 60 (1 minute)
# - Set RAG_CONFIG["max_cache_entries"] = 10
# - Set RAG_CONFIG["deduplication_window"] = 5
#
# No Caching (for testing/development):
# - Set RAG_CONFIG["vector_cache_enabled"] = False
# - Set RAG_CONFIG["deduplication_window"] = 0
#
# CONTEXT ENHANCEMENT SCENARIOS:
# =============================
#
# New User (no conversation history):
# - Set RAG_CONFIG["enable_conversation_context"] = False
# - Set RAG_CONFIG["enable_knowledge_context"] = True
# - Set RAG_CONFIG["max_knowledge_chars"] = 600 (more knowledge context)
#
# Returning User (rich conversation history):
# - Set RAG_CONFIG["enable_conversation_context"] = True
# - Set RAG_CONFIG["max_conversation_exchanges"] = 3
# - Set RAG_CONFIG["enable_knowledge_context"] = True
# - Set RAG_CONFIG["max_knowledge_chars"] = 400 (balance knowledge/conversation)
#
# Crisis Intervention (focus on immediate context):
# - Set RAG_CONFIG["enable_conversation_context"] = True
# - Set RAG_CONFIG["max_conversation_exchanges"] = 5 (full recent history)
# - Set RAG_CONFIG["enable_knowledge_context"] = False (focus on conversation)
# - Set RAG_CONFIG["similarity_threshold"] = 0.9 (only perfect matches)
#
# General Support (balanced approach):
# - Set RAG_CONFIG["enable_conversation_context"] = True
# - Set RAG_CONFIG["enable_knowledge_context"] = True
# - Set RAG_CONFIG["max_conversation_exchanges"] = 2
# - Set RAG_CONFIG["max_knowledge_chars"] = 500
# - Set RAG_CONFIG["similarity_threshold"] = 0.7
#
# TROUBLESHOOTING RAG PERFORMANCE:
# ===============================
#
# If responses are too slow:
# 1. Reduce RAG_CONFIG["max_knowledge_chars"] to 300
# 2. Enable RAG_CONFIG["vector_cache_enabled"] = True
# 3. Increase RAG_CONFIG["deduplication_window"] to 10
# 4. Disable RAG_CONFIG["enable_associative_memory"]
#
# If responses lack context:
# 1. Increase RAG_CONFIG["max_knowledge_chars"] to 800
# 2. Lower RAG_CONFIG["similarity_threshold"] to 0.6
# 3. Increase RAG_CONFIG["max_conversation_exchanges"] to 4
# 4. Enable all RAG_CONFIG["enable_*"] features
#
# If memory usage is high:
# 1. Reduce RAG_CONFIG["max_cache_entries"] to 5
# 2. Lower RAG_CONFIG["similarity_cache_timeout"] to 30
# 3. Disable RAG_CONFIG["enable_associative_memory"]
# 4. Reduce RAG_CONFIG["max_knowledge_chars"] to 250
#
# If context quality is poor:
# 1. Increase RAG_CONFIG["similarity_threshold"] to 0.8
# 2. Enable detailed logging to debug retrieval
# 3. Check vector database content quality
# 4. Verify embedding model compatibility
