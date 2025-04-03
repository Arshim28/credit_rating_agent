import numpy as np
import time
from typing import List, Optional, Dict
from src.rag_components.utils import log_memory_usage, log_array_info, logger, WatchdogTimer
from src.rag_components.config import GOOGLE_API_KEY, DEFAULT_CONFIG
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_result, RetryError

class EmbeddingProvider:
    """Base class for embedding providers to ensure consistent interface"""
    
    def __init__(self, api_key: str, model_name: str, dimension: Optional[int] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_dimension = dimension
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_single_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text"""
        raise NotImplementedError("Subclasses must implement this method")


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Google's Gemini API"""
    
    def __init__(self, api_key: str = GOOGLE_API_KEY, model_name: str = "text-embedding-004", 
                 dimension: Optional[int] = DEFAULT_CONFIG["embedding_dimension"]):
        super().__init__(api_key, model_name, dimension)
        self.max_tokens = DEFAULT_CONFIG["max_tokens"]
        self.retry_max_attempts = DEFAULT_CONFIG["retry_max_attempts"]
        self.retry_base_delay = DEFAULT_CONFIG["retry_base_delay"]
        self.request_delay = DEFAULT_CONFIG["request_delay"]
        
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=self.api_key)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using batching for efficiency.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), embedding_dimension)
        """
        logger.warning(f"START EMBEDDING: Generating embeddings for {len(texts)} texts, total text length: {sum(len(t) for t in texts)}")
        log_memory_usage("before_embeddings")
        
        # Use true batching when possible (for shorter texts)
        # For longer texts, we'll process them individually
        
        # We'll divide texts into smaller batches to avoid rate limits
        max_batch_size = 20  # Adjust based on API limits
        all_embeddings = []
        failed_indices = []
        
        # Divide texts into batches
        for batch_start in range(0, len(texts), max_batch_size):
            batch_end = min(batch_start + max_batch_size, len(texts))
            batch = texts[batch_start:batch_end]
            
            if batch_start > 0:
                time.sleep(self.request_delay * 2)  # Longer delay between batches
                
            logger.warning(f"PROCESSING BATCH {batch_start//max_batch_size + 1}/{(len(texts) + max_batch_size - 1)//max_batch_size}")
            log_memory_usage(f"before_batch_{batch_start//max_batch_size + 1}")
            
            # Try to batch process if all texts are under a certain length
            if all(len(text) < 10000 for text in batch):
                try:
                    logger.info(f"Attempting batch processing for {len(batch)} texts")
                    batch_embeddings = self._get_batch_embeddings_with_retry(batch)
                    all_embeddings.append(batch_embeddings)
                    logger.info(f"Successfully batch processed {len(batch)} texts")
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}. Falling back to individual processing.")
                    # If batch fails, process each text individually
                    batch_embeddings, batch_failed = self._process_texts_individually(batch, batch_start)
                    all_embeddings.append(batch_embeddings)
                    failed_indices.extend(batch_failed)
            else:
                # Process texts individually if they're too long for batching
                logger.info(f"Using individual processing for {len(batch)} texts (some texts too long)")
                batch_embeddings, batch_failed = self._process_texts_individually(batch, batch_start)
                all_embeddings.append(batch_embeddings)
                failed_indices.extend(batch_failed)
                
            log_memory_usage(f"after_batch_{batch_start//max_batch_size + 1}")
        
        if failed_indices:
            logger.error(f"Failed to get embeddings for {len(failed_indices)} texts at indices: {failed_indices}")
        
        # Concatenate all batch results
        if len(all_embeddings) > 1:
            logger.warning("CONCATENATING EMBEDDING BATCHES")
            log_memory_usage("before_concat_embeddings")
            result = np.concatenate(all_embeddings, axis=0)
            log_memory_usage("after_concat_embeddings")
        else:
            result = all_embeddings[0]
        
        # Update embedding dimension if needed
        if self.embedding_dimension is None and result.shape[1]:
            self.embedding_dimension = result.shape[1]
            logger.info(f"Setting embedding dimension to {self.embedding_dimension} based on results")
        
        log_array_info("embeddings_result", result)
        log_memory_usage("after_embeddings")
        
        return result

    def _process_texts_individually(self, texts: List[str], base_index: int = 0) -> tuple[np.ndarray, List[int]]:
        """Process a list of texts individually and return as a batch."""
        embeddings = []
        failed_indices = []
        
        for i, text in enumerate(texts):
            idx = base_index + i
            if i > 0:
                time.sleep(self.request_delay)
                
            # Set a watchdog timer for each embedding request
            watchdog = WatchdogTimer(timeout=45, operation_name=f"Embedding request {idx+1}/{base_index+len(texts)}")
            watchdog.start()
            
            try:
                embedding = self._get_single_embedding_with_retry(text)
                embeddings.append(embedding)
            except RetryError as e:
                logger.error(f"Failed to get embedding for text index {idx} after all retries: {str(e.__cause__)}")
                failed_indices.append(idx)
                
                # Set embedding dimension from successful embeddings if available
                if embeddings and self.embedding_dimension is None:
                    self.embedding_dimension = len(embeddings[0])
                
                # Create a zero vector for failed embedding
                if self.embedding_dimension is not None:
                    logger.warning(f"Using zero vector for failed embedding with dimension {self.embedding_dimension}")
                    zero_embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                    embeddings.append(zero_embedding)
            finally:
                watchdog.stop()
        
        return np.array(embeddings), failed_indices
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=(
            retry_if_exception_type(TimeoutError) | 
            retry_if_exception_type(ConnectionError) |
            retry_if_exception_type(Exception)
        ),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying embedding request in {retry_state.next_action.sleep} seconds (attempt {retry_state.attempt_number})"
        )
    )
    def _get_batch_embeddings_with_retry(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts with retry logic."""
        if not texts:
            return np.array([])
            
        config = None
        if self.embedding_dimension is not None:
            config = self.types.EmbedContentConfig(
                output_dimensionality=self.embedding_dimension
            )

        logger.info(f"Generating batch embeddings for {len(texts)} texts")
        
        # Set a watchdog timer for the batch request
        watchdog = WatchdogTimer(timeout=90, operation_name=f"Batch embedding request for {len(texts)} texts")
        watchdog.start()
        
        try:
            # The Python client handles lists of texts correctly in embed_content
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,  # Pass the list directly
                config=config
            )
            
            # Process results - when a list is passed, result.embeddings is a list
            embeddings = []
            for embedding_result in result.embeddings:
                embedding = np.array(embedding_result.values, dtype=np.float32)
                # Normalize the embedding
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            # Update dimension based on actual result
            if embeddings and self.embedding_dimension is None:
                self.embedding_dimension = len(embeddings[0])
                logger.info(f"Set embedding dimension to {self.embedding_dimension}")
            
            return np.array(embeddings)
            
        except Exception as e:
            is_rate_limit = "429" in str(e)
            is_server_error = any(code in str(e) for code in ["500", "502", "503", "504"])
            is_timeout = "timeout" in str(e).lower() or "timed out" in str(e).lower()
            
            if is_rate_limit:
                logger.error(f"Rate limit exceeded: {e}")
                raise ConnectionError(f"Rate limit exceeded: {e}")
            elif is_server_error:
                logger.error(f"Server error: {e}")
                raise ConnectionError(f"Server error: {e}")
            elif is_timeout:
                logger.error(f"Timeout error: {e}")
                raise TimeoutError(f"Request timed out: {e}")
            else:
                logger.error(f"Unknown error generating embedding: {e}")
                raise Exception(f"Unknown error: {e}")
        finally:
            watchdog.stop()
            
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=(
            retry_if_exception_type(TimeoutError) | 
            retry_if_exception_type(ConnectionError) |
            retry_if_exception_type(Exception)
        ),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying embedding request in {retry_state.next_action.sleep} seconds (attempt {retry_state.attempt_number})"
        )
    )
    def _get_single_embedding_with_retry(self, text: str) -> np.ndarray:
        """Get embedding for a single text with retry logic."""
        return self._get_single_embedding(text)
    
    def _get_single_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        config = None
        if self.embedding_dimension is not None:
            config = self.types.EmbedContentConfig(
                output_dimensionality=self.embedding_dimension
            )

        logger.info(f"Generating embedding for text: {text[:100]}...")
        
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=config
            )

            embedding = np.array(result.embeddings[0].values, dtype=np.float32)
            
            # Always update dimension based on actual result
            self.embedding_dimension = embedding.shape[0]
            
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
            
        except Exception as e:
            is_rate_limit = "429" in str(e)
            is_server_error = any(code in str(e) for code in ["500", "502", "503", "504"])
            is_timeout = "timeout" in str(e).lower() or "timed out" in str(e).lower()
            
            if is_rate_limit:
                logger.error(f"Rate limit exceeded: {e}")
                raise ConnectionError(f"Rate limit exceeded: {e}")
            elif is_server_error:
                logger.error(f"Server error: {e}")
                raise ConnectionError(f"Server error: {e}")
            elif is_timeout:
                logger.error(f"Timeout error: {e}")
                raise TimeoutError(f"Request timed out: {e}")
            else:
                logger.error(f"Unknown error generating embedding: {e}")
                raise Exception(f"Unknown error: {e}")