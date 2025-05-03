"""
associative_memory.py

This module implements a semantic associative memory system for psychological counseling applications.
The AssociativeMemory class provides capabilities to store, retrieve and associate memories based on
semantic similarity and topic connections, enabling a rich network of psychological concepts.

Key Features:
- Semantic similarity search using FAISS vector database
- Topic-based memory associations
- Hybrid retrieval combining vector similarity and topic relationships
- Fallback mechanisms for robustness when embedding models are unavailable
- Memory caching for performance optimization

Technical Implementation:
- Uses sentence transformer models for semantic embedding generation
- Employs FAISS for efficient vector similarity search
- Provides graceful degradation to keyword-based search as fallback
- Includes caching to optimize repeated queries

Classes:
- AssociativeMemory: The main class that manages semantic memories and their associations

Dependencies:
- faiss: For efficient vector similarity search
- numpy: For numerical operations
- ModelManager: For consistent embedding generation across the application
"""

import traceback
from typing import Any, Dict, List, Optional, Set

import numpy as np
from faiss import Index, IndexFlatIP, normalize_L2
from prismalog.log import get_logger

# Import the ModelManager
from psy_supabase.core.model_manager import get_embedding_provider

logger = get_logger(__name__)


class AssociativeMemory:
    """
    Associative Memory for psychological counseling that makes connections between related concepts.
    Uses semantic similarity and explicit associations to create a rich memory network.

    The memory system stores content with topic tags and retrieves related information using:
    1. Direct semantic similarity via vector embeddings
    2. Associated memories connected through shared topics

    Attributes:
        memories: List of memory objects containing content, topics, and embeddings
        memory_embeddings: Numpy array of all memory embeddings
        index: FAISS index for efficient similarity search
        memory_index_map: Mapping from FAISS indices to memory indices
        topics_to_memories: Dictionary mapping topics to memory indices
        cache: Query cache for performance optimization
        model_name: Name of the embedding model being used
        dimension: Dimensionality of the embeddings
    """

    memories: List[Dict[str, Any]]
    memory_embeddings: Optional[np.ndarray]
    index: Optional[Index]  # Use the base Index type or IndexFlatIP specifically
    memory_index_map: Dict[int, int]
    topics_to_memories: Dict[str, Set[int]]
    cache: Dict[str, List[str]]
    model_name: str
    dimension: int
    embedding_provider: Optional[Any]  # Or a more specific provider type if available

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initialize the associative memory with a semantic model.

        Args:
            model_name: Name of the transformer model to use for embeddings
                       (default: "sentence-transformers/all-MiniLM-L6-v2")
        """
        self.memories: List[Dict[str, Any]] = []
        self.memory_embeddings = None
        self.index = None
        self.memory_index_map: Dict[int, int] = {}
        self.topics_to_memories: Dict[str, Set[int]] = {}
        self.cache: Dict[str, List[str]] = {}
        self.model_name = model_name

        # Use ModelManager instead of direct model loading
        try:
            # Initialize the embedding provider through model manager
            self.embedding_provider = get_embedding_provider(model_name=model_name)
            if self.embedding_provider:
                self.dimension = self.embedding_provider.get_embedding_dimension()
                logger.info("AssociativeMemory initialized with model: %s (dim: %d)", model_name, self.dimension)
            else:
                # Handle case where provider failed to load immediately
                self.dimension = 384  # Default fallback dimension
                logger.error("Failed to initialize embedding provider. Using fallback dimension %d.", self.dimension)

        except Exception as e:
            logger.error("Error loading embedding model: %s", e)
            # Fall back to a simple word-based similarity
            self.embedding_provider = None
            self.dimension = 384  # Default fallback dimension
            logger.warning("Using fallback word-based similarity")

    def add_memory(self, content: str, topics: List[str], metadata: Optional[Dict] = None) -> int:
        """
        Add a memory to the associative store with topics for connections.

        Args:
            content: The memory content text
            topics: List of topics/tags to associate with this memory
            metadata: Optional metadata for the memory

        Returns:
            The index of the newly added memory
        """
        try:
            # Create memory object
            memory_obj = {
                "content": content,
                "topics": topics,
                "metadata": metadata or {},
                "embedding": None,
                "index": len(self.memories),
            }

            # Generate embedding
            embedding = self._generate_embedding(content)
            memory_obj["embedding"] = embedding

            self.memories.append(memory_obj)
            memory_index = len(self.memories) - 1

            # Associate with topics
            for topic in topics:
                topic_lower = topic.lower()
                if topic_lower not in self.topics_to_memories:
                    self.topics_to_memories[topic_lower] = set()
                self.topics_to_memories[topic_lower].add(memory_index)

            # Update the FAISS index
            self._update_index()

            # Invalidate cache for related topics
            for topic in topics:
                topic_lower = topic.lower()
                for cached_query in list(self.cache.keys()):
                    if topic_lower in cached_query.lower():
                        del self.cache[cached_query]

            return memory_index

        except Exception as e:
            logger.error("Error adding memory: %s", e)
            return -1

    def retrieve_memories(self, query_text: str, top_k: int = 5, threshold: float = 0.6) -> List[str]:
        """
        Query the associative memory for relevant memories.

        The query process:
        1. Checks the cache for previously seen queries
        2. Performs semantic similarity search using vector embeddings
        3. Finds associated memories through shared topic connections using _find_associations
        4. Falls back to keyword search if semantic search fails
        5. Formats and returns the combined results

        Args:
            query_text: The query text to search for
            top_k: Maximum number of direct results to return (default: 5)
            threshold: Similarity threshold for direct results (default: 0.6)

        Returns:
            List of memory contents, including associated memories with their relevance scores
        """
        try:
            # Check cache first
            if query_text in self.cache:
                logger.info("Cache hit for query: %s...", query_text[:50])
                cached_result = self.cache[query_text]
                if isinstance(cached_result, list):
                    return cached_result
                else:
                    logger.warning("Cache contained unexpected format for key %s. Re-fetching.", query_text[:50])

            if not self.memories:
                return ["No memories available."]

            # Get query embedding
            query_embedding = self._generate_embedding(query_text)

            # Perform semantic search if index exists
            direct_results: List[Dict[str, Any]] = []
            if self.index is not None and query_embedding is not None:
                # Normalize the query embedding for cosine similarity
                query_embedding_norm = np.array([query_embedding], dtype=np.float32)
                normalize_L2(query_embedding_norm)

                # Search the index - for IndexFlatIP, higher scores are better
                S, I = self.index.search(query_embedding_norm, top_k)

                # Process results
                for score, idx in zip(S[0], I[0]):
                    # Skip if below threshold or invalid index
                    if score < threshold or idx < 0:
                        continue

                    # Map FAISS index to memory object index
                    memory_idx = self.memory_index_map.get(int(idx), -1)
                    if memory_idx == -1 or memory_idx >= len(self.memories):
                        continue

                    memory = self.memories[memory_idx]
                    direct_results.append(
                        {
                            "content": memory["content"],
                            "topics": memory["topics"],
                            "similarity": float(score),
                            "index": memory_idx,
                            "is_direct": True,
                        }
                    )

            # If no semantic results or no index, fall back to keyword matching
            if not direct_results:
                logger.info("No semantic results found or index unavailable, using keyword matching")
                direct_results = self._keyword_search(query_text, top_k)

            # Find associated memories
            associated_memories = self._find_associations(direct_results, query_text)

            # Combine direct results and associated memories
            all_results = direct_results + associated_memories

            # Sort by similarity (or relevance)
            all_results.sort(key=lambda x: x["similarity"], reverse=True)

            # Format the results
            formatted_results: List[str] = []
            added_indices: Set[int] = set()
            for result in all_results:
                result_index = result.get("index")
                if not isinstance(result_index, int):
                    logger.warning("Skipping result with non-integer index: %s", result_index)
                    continue

                if result_index in added_indices:
                    continue

                content = result.get("content", "")
                if not isinstance(content, str):
                    logger.warning("Skipping result with non-string content: %s", type(content))
                    continue

                if result.get("is_direct", False):
                    formatted_results.append(content)
                else:
                    similarity = result.get("similarity", 0.0)
                    formatted_results.append(f"{content} (Associated Memory, Relevance: {similarity:.2f})")
                added_indices.add(result_index)

            # Cache the result
            self.cache[query_text] = formatted_results

            return formatted_results

        except Exception as e:
            logger.error("Error querying associative memory: %s", e)
            logger.error(traceback.format_exc())
            return ["Error retrieving memories."]

    def clear_cache(self) -> None:
        """
        Clear the query cache.

        This removes all cached query results, forcing future queries
        to perform a full search against the memory store.
        """
        self.cache = {}
        logger.info("Associative memory cache cleared")

    def get_topics(self) -> List[str]:
        """
        Get all topics in the memory store.

        Returns:
            List of all unique topics across all memories
        """
        return list(self.topics_to_memories.keys())

    def get_memories_by_topic(self, topic: str) -> List[Dict]:
        """
        Get all memories associated with a topic.

        Args:
            topic: The topic to search for (case-insensitive)

        Returns:
            List of memory objects that are tagged with the specified topic
        """
        topic_lower = topic.lower()
        memory_indices = self.topics_to_memories.get(topic_lower, set())
        return [self.memories[idx] for idx in memory_indices if idx < len(self.memories)]

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text using ModelManager.

        Uses the configured embedding model to generate a vector representation
        of the input text. Falls back to a simple embedding if the model fails.

        Args:
            text: The text to generate an embedding for

        Returns:
            Numpy array containing the embedding vector
        """
        if self.embedding_provider is not None:
            try:
                # Get embedding as a list of floats
                embedding_list = self.embedding_provider.generate_embedding(text)
                # Convert to numpy array
                return np.array(embedding_list, dtype=np.float32)
            except Exception as e:
                logger.error("Error generating embedding with ModelManager: %s", e)
                # Fall back to simple embedding
                return self._simple_embedding(text)
        else:
            # Use simple word vector if no model available
            return self._simple_embedding(text)

    def _simple_embedding(self, text: str) -> np.ndarray:
        """
        Generate a simple embedding based on word presence.

        This is a fallback method used when the main embedding model fails.
        It creates a basic word-presence vector using hashing.

        Args:
            text: The text to generate a simple embedding for

        Returns:
            Numpy array containing the simple embedding vector
        """
        # Very simple embedding - just for fallback
        words = text.lower().split()
        vec = np.zeros(self.dimension)
        for _, word in enumerate(words):
            hash_val = hash(word) % self.dimension
            vec[hash_val % self.dimension] += 1.0

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _update_index(self) -> None:
        """
        Update the FAISS index with all memory embeddings.

        This rebuilds the entire FAISS index whenever new memories are added,
        enabling efficient similarity search across all memories.
        """
        try:
            # Check if there are any embeddings to add
            valid_embeddings = [m["embedding"] for m in self.memories if m.get("embedding") is not None]
            if not valid_embeddings:
                logger.warning("No valid embeddings found in memories. Cannot update index.")
                self.index = None  # Ensure index is None if no embeddings
                self.memory_index_map = {}
                return

            embeddings = np.array(valid_embeddings, dtype=np.float32)

            # Check if embeddings array is empty after filtering
            if embeddings.size == 0:
                logger.warning("Embeddings array is empty after filtering None values. Cannot update index.")
                self.index = None
                self.memory_index_map = {}
                return

            # Normalize embeddings for cosine similarity
            normalize_L2(embeddings)

            # Create new index - use IndexFlatIP for cosine similarity
            new_index = IndexFlatIP(embeddings.shape[1])
            # The add method is on the index object itself, not None
            new_index.add(x=embeddings)  # pylint: disable=no-value-for-parameter

            # Update the index and mapping only after successful creation/add
            self.index = new_index
            # Rebuild map based on memories that had valid embeddings
            self.memory_index_map = {
                faiss_idx: mem_idx
                for faiss_idx, (mem_idx, mem) in enumerate(enumerate(self.memories))
                if mem.get("embedding") is not None
            }

            logger.info(
                "Updated associative memory index with %d memories (cosine similarity)",
                self.index.ntotal if self.index else 0,
            )

        except Exception as e:
            logger.error("Error updating FAISS index: %s", e)
            logger.error(traceback.format_exc())  # Log traceback for detailed error
            self.index = None  # Ensure index is None on error
            self.memory_index_map = {}  # Clear map on error

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Fallback keyword-based search for when vector search is unavailable.

        Performs a simple word overlap calculation between the query and memory contents.

        Args:
            query: The query text
            top_k: Maximum number of results to return

        Returns:
            List of memory dictionaries sorted by word overlap similarity
        """
        results = []
        query_words = set(query.lower().split())

        for idx, memory in enumerate(self.memories):
            content_words = set(memory["content"].lower().split())
            overlap = len(query_words.intersection(content_words))

            # Calculate simple similarity based on word overlap
            similarity = overlap / max(1, len(query_words))

            if similarity > 0:
                results.append(
                    {
                        "content": memory["content"],
                        "topics": memory["topics"],
                        "similarity": similarity,
                        "index": idx,
                        "is_direct": True,
                    }
                )

        # Sort and limit
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _find_associations(self, direct_results: List[Dict], query_text: str) -> List[Dict]:
        """
        Find associated memories based on topics from direct results.

        Discovers second-order connections by finding memories that share topics
        with the direct search results but weren't included in those results.

        Args:
            direct_results: List of direct search result dictionaries
            query_text: The original query text (used to extract potential topics)

        Returns:
            List of associated memory dictionaries
        """
        if not direct_results:
            return []

        # Collect topics from direct results
        topics = set()
        for result in direct_results:
            topics.update([t.lower() for t in result["topics"]])

        # Extract potential topics from query text itself
        query_words = set(query_text.lower().split())
        for topic in self.topics_to_memories:
            # Check if the topic (which might be multi-word) is present in the query
            if topic in query_text.lower():  # Simple substring check, could be improved
                topics.add(topic)
            # Or check if individual words of the topic are in the query
            elif all(word in query_words for word in topic.split()):
                topics.add(topic)

        # Find memories associated with these topics
        association_candidates = set()
        for topic in topics:
            if topic in self.topics_to_memories:
                association_candidates.update(self.topics_to_memories[topic])

        # Remove direct result indices
        direct_indices = {result["index"] for result in direct_results}
        association_candidates = association_candidates - direct_indices

        # Get associated memories
        associated_results = []
        for idx in association_candidates:
            if idx >= len(self.memories):
                continue

            memory = self.memories[idx]

            # Calculate association strength based on topic overlap
            memory_topics_lower = {t.lower() for t in memory["topics"]}
            topic_overlap = len(memory_topics_lower.intersection(topics))

            # Define association strength (adjust logic as needed)
            # Example: base score + bonus for overlap
            association_strength = min(0.8, 0.6 + 0.05 * topic_overlap)

            associated_results.append(
                {
                    "content": memory["content"],
                    "topics": memory["topics"],
                    "similarity": association_strength,  # Use calculated strength
                    "index": idx,
                    "is_direct": False,
                }
            )

        return associated_results
