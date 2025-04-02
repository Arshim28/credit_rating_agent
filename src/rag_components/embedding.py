import numpy as np
import time
from typing import List, Optional
from utils import log_memory_usage, log_array_info, logger, WatchdogTimer
from config import GOOGLE_API_KEY, DEFAULT_CONFIG
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_result, RetryError

class GeminiEmbeddingProvider:
    def __init__(self, api_key: str = GOOGLE_API_KEY, model_name: str = "text-embedding-004", 
                 dimension: Optional[int] = DEFAULT_CONFIG["embedding_dimension"]):
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_dimension = dimension
        self.max_tokens = DEFAULT_CONFIG["max_tokens"]
        self.retry_max_attempts = DEFAULT_CONFIG["retry_max_attempts"]
        self.retry_base_delay = DEFAULT_CONFIG["retry_base_delay"]
        self.request_delay = DEFAULT_CONFIG["request_delay"]
        
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=self.api_key)
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        logger.warning(f"START EMBEDDING: Generating embeddings for {len(texts)} texts, total text length: {sum(len(t) for t in texts)}")
        log_memory_usage("before_embeddings")
        
        embeddings = []
        failed_indices = []
        
        for i, text in enumerate(texts):
            if i > 0 and i % 10 == 0:
                logger.warning(f"EMBEDDING PROGRESS: {i}/{len(texts)} texts processed")
                log_memory_usage(f"embedding_batch_{i}")
                
            if i > 0:
                time.sleep(self.request_delay)
                
            # Set a watchdog timer for each embedding request
            watchdog = WatchdogTimer(timeout=45, operation_name=f"Embedding request {i+1}/{len(texts)}")
            watchdog.start()
            
            try:
                try:
                    embedding = self._get_single_embedding_with_retry(text)
                    embeddings.append(embedding)
                except RetryError as e:
                    logger.error(f"Failed to get embedding for text index {i} after all retries: {str(e.__cause__)}")
                    failed_indices.append(i)
                    
                    try:
                        embedding = self._get_single_embedding_with_batch_retry(text)
                        embeddings.append(embedding)
                        failed_indices.pop()
                    except RetryError as batch_e:
                        logger.error(f"All batch-level retries failed for text index {i}: {str(batch_e.__cause__)}")
                        if len(embeddings) > 0 and self.embedding_dimension is None:
                            self.embedding_dimension = len(embeddings[0])
                        
                        if self.embedding_dimension is not None:
                            logger.warning(f"Using zero vector for failed embedding with dimension {self.embedding_dimension}")
                            zero_embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                            embeddings.append(zero_embedding)
            finally:
                watchdog.stop()
        
        if failed_indices:
            logger.error(f"Failed to get embeddings for {len(failed_indices)} texts at indices: {failed_indices}")
        
        result = np.array(embeddings)
        
        log_array_info("embeddings_result", result)
        log_memory_usage("after_embeddings")
        
        return result
    
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
        return self._get_single_embedding(text)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, min=5, max=120),
        retry=(
            retry_if_exception_type(TimeoutError) | 
            retry_if_exception_type(ConnectionError) |
            retry_if_exception_type(Exception)
        ),
        before_sleep=lambda retry_state: logger.warning(
            f"Batch-level retry in {retry_state.next_action.sleep} seconds (attempt {retry_state.attempt_number})"
        )
    )
    def _get_single_embedding_with_batch_retry(self, text: str) -> np.ndarray:
        return self._get_single_embedding(text)
    
    def _get_single_embedding(self, text: str) -> np.ndarray:
        config = None
        if self.embedding_dimension is not None:
            config = self.types.EmbedContentConfig(
                output_dimensionality=self.embedding_dimension
            )

        logger.info(f"Generating embedding for text: {text[:100]}...")
        logger.info(f"Config: {config}")
        
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=config
            )

            embedding = np.array(result.embeddings[0].values, dtype=np.float32)
            
            # Always update dimension based on actual result
            self.embedding_dimension = embedding.shape[0]
            logger.info(f"Set embedding dimension to {self.embedding_dimension}")
            
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