import os
import gc
import logging
from typing import Dict, Optional
from src.rag_components.ocr_vector_store import OCRVectorStore
from src.rag_components.utils import log_memory_usage, logger

class VectorStoreManager:
    """
    Manages vector stores with an LRU (Least Recently Used) caching mechanism
    to keep memory usage under control.
    """
    
    def __init__(self, vector_store_dir: str, max_stores: int = 3):
        """
        Initialize the vector store manager.
        
        Args:
            vector_store_dir: Directory where vector stores are saved
            max_stores: Maximum number of vector stores to keep in memory
        """
        self.vector_store_dir = vector_store_dir
        self.max_stores = max_stores
        self.stores: Dict[str, OCRVectorStore] = {}
        self.lru = []  # Least recently used tracking
        logger.warning(f"Initialized VectorStoreManager with max_stores={max_stores}")
        
    def get_store(self, path: str) -> Optional[OCRVectorStore]:
        """
        Get a vector store, loading it from disk if necessary.
        
        Args:
            path: Path to the report PDF (used as the key)
            
        Returns:
            OCRVectorStore: The loaded vector store, or None if not found
        """
        logger.info(f"Requesting vector store for {path}")
        
        # If already in memory, update LRU and return
        if path in self.stores:
            logger.info(f"Found {path} in memory cache")
            self._update_lru(path)
            return self.stores[path]
        
        # Otherwise, try to load from disk
        store_dir = self._get_store_dir(path)
        if not os.path.exists(store_dir) or not os.path.isdir(store_dir):
            logger.warning(f"No vector store directory found for {path}")
            return None
            
        log_memory_usage(f"before_load_{os.path.basename(path)}")
        logger.info(f"Loading vector store from {store_dir}")
        
        try:
            store = OCRVectorStore()
            store.load(store_dir)
            self._add_store(path, store)
            log_memory_usage(f"after_load_{os.path.basename(path)}")
            return store
        except Exception as e:
            logger.error(f"Error loading vector store for {path}: {e}")
            return None
    
    def create_store(self, path: str, chunk_size: int = None, chunk_overlap: int = None, 
                     index_type: str = None) -> OCRVectorStore:
        """
        Create a new vector store for a report.
        
        Args:
            path: Path to the report PDF
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            index_type: Optional index type override
            
        Returns:
            OCRVectorStore: The newly created vector store
        """
        logger.info(f"Creating new vector store for {path}")
        log_memory_usage(f"before_create_{os.path.basename(path)}")
        
        # Create the store with custom parameters if provided
        kwargs = {}
        if chunk_size is not None:
            kwargs['chunk_size'] = chunk_size
        if chunk_overlap is not None:
            kwargs['chunk_overlap'] = chunk_overlap
        if index_type is not None:
            kwargs['index_type'] = index_type
            
        store = OCRVectorStore(**kwargs)
        
        # Add to our managed stores
        self._add_store(path, store)
        
        log_memory_usage(f"after_create_{os.path.basename(path)}")
        return store
    
    def save_store(self, path: str) -> bool:
        """
        Save a vector store to disk.
        
        Args:
            path: Path to the report PDF
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if path not in self.stores:
            logger.warning(f"Cannot save non-existent store for {path}")
            return False
            
        store_dir = self._get_store_dir(path)
        os.makedirs(store_dir, exist_ok=True)
        
        logger.info(f"Saving vector store to {store_dir}")
        log_memory_usage(f"before_save_{os.path.basename(path)}")
        
        try:
            self.stores[path].save(store_dir)
            log_memory_usage(f"after_save_{os.path.basename(path)}")
            return True
        except Exception as e:
            logger.error(f"Error saving vector store for {path}: {e}")
            return False
    
    def process_document(self, path: str) -> Optional[OCRVectorStore]:
        """
        Process a document, creating a vector store and saving it.
        
        Args:
            path: Path to the report PDF
            
        Returns:
            OCRVectorStore: The processed vector store, or None if processing failed
        """
        logger.warning(f"PROCESSING DOCUMENT: {path}")
        
        # Check if already processed and stored on disk
        store_dir = self._get_store_dir(path)
        if os.path.exists(store_dir) and os.path.isdir(store_dir) and os.listdir(store_dir):
            logger.info(f"Document already processed, loading from {store_dir}")
            return self.get_store(path)
            
        # Create a new store
        store = self.create_store(
            path,
            chunk_size=5000,  # Optimal for rating reports
            chunk_overlap=500,
            index_type="HNSW"  # Best performance/accuracy tradeoff
        )
        
        # Process the document
        log_memory_usage(f"before_process_{os.path.basename(path)}")
        try:
            store.add_document(path)
            log_memory_usage(f"after_process_{os.path.basename(path)}")
            
            # Save to disk
            self.save_store(path)
            
            return store
        except Exception as e:
            logger.error(f"Error processing document {path}: {e}")
            # Remove from managed stores since processing failed
            if path in self.stores:
                del self.stores[path]
                if path in self.lru:
                    self.lru.remove(path)
            return None
    
    def _get_store_dir(self, path: str) -> str:
        """Generate a directory path for storing a vector store."""
        base_name = os.path.basename(path)
        parent_dir = os.path.basename(os.path.dirname(path))
        return os.path.join(self.vector_store_dir, f"{parent_dir}_{base_name}")
    
    def _add_store(self, path: str, store: OCRVectorStore):
        """Add a store to managed stores, evicting LRU if necessary."""
        # If at capacity, evict the least recently used store
        if len(self.stores) >= self.max_stores:
            self._evict_lru()
        
        # Add the new store and update LRU
        self.stores[path] = store
        self._update_lru(path)
        logger.info(f"Added {path} to managed stores (total: {len(self.stores)})")
    
    def _evict_lru(self):
        """Evict the least recently used store from memory."""
        if not self.lru:
            return
            
        lru_path = self.lru.pop(0)
        logger.warning(f"EVICTING LRU vector store: {lru_path}")
        log_memory_usage("before_eviction")
        
        # Ensure store is saved before eviction
        if lru_path in self.stores:
            try:
                self.save_store(lru_path)
            except Exception as e:
                logger.error(f"Error saving store before eviction: {e}")
            
            del self.stores[lru_path]
            gc.collect()
            
        log_memory_usage("after_eviction")
    
    def _update_lru(self, path: str):
        """Update the LRU tracking for a path."""
        # Remove if already in list
        if path in self.lru:
            self.lru.remove(path)
        
        # Add to end (most recently used)
        self.lru.append(path)
        
    def close_all(self):
        """Save and close all vector stores."""
        logger.warning(f"Closing all vector stores ({len(self.stores)})")
        
        for path in list(self.stores.keys()):
            try:
                self.save_store(path)
            except Exception as e:
                logger.error(f"Error saving store during close: {e}")
                
        self.stores.clear()
        self.lru.clear()
        gc.collect()
        
    def __del__(self):
        """Ensure all stores are saved when the manager is deleted."""
        try:
            self.close_all()
        except:
            pass