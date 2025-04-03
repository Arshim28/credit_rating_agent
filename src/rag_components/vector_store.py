import os
import json
import gc
import numpy as np
import faiss
from typing import List, Tuple
from src.rag_components.text_chunk import TextChunk
from src.rag_components.utils import log_memory_usage, log_array_info, logger, chunk_generator

class FaissVectorStore:
    def __init__(self, dimension: int = None, index_type: str = "Flat"):
        self.dimension = dimension  # Can be None initially
        self.index_type = index_type
        self.index = None
        self.chunks = []
        
    def init_index(self, dimension=None):
        if dimension is not None:
            self.dimension = dimension
            
        if self.dimension is None:
            raise ValueError("Cannot initialize FAISS index: dimension not set")
            
        logger.warning(f"INITIALIZING FAISS INDEX: type={self.index_type}, dimension={self.dimension}")
        log_memory_usage("before_index_init")
        
        if self.index_type == "Flat":
            flat_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(flat_index)
            logger.warning("Initialized Flat index")
        elif self.index_type == "IVF":
            nlist = 100
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "IVFPQ":
            nlist = 100
            m = 16
            bits = 8
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, bits)
        elif self.index_type == "HNSW":
            M = 32
            hnsw_index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
            self.index = faiss.IndexIDMap(hnsw_index)
            logger.warning("Initialized HNSW index")
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
            
        log_memory_usage("after_index_init")
            
    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray):
        logger.warning(f"ADDING CHUNKS TO INDEX: {len(chunks)} chunks, embeddings shape: {embeddings.shape}")
        log_memory_usage("before_add_chunks")
        log_array_info("embeddings_to_add", embeddings)
        
        # Get dimension from the embeddings if not already set
        if self.dimension is None and embeddings.shape[1]:
            self.dimension = embeddings.shape[1]
            logger.warning(f"Setting vector store dimension to {self.dimension} from embeddings")
        
        if not self.index:
            self.init_index(dimension=embeddings.shape[1])
            
        max_batch_size = 100
        
        if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:
            logger.warning(f"TRAINING IVF INDEX with {len(embeddings)} vectors")
            log_memory_usage("before_index_training")
            self.index.train(embeddings)
            log_memory_usage("after_index_training")
            
        start_id = len(self.chunks)
        
        if len(chunks) > max_batch_size:
            logger.warning(f"BATCH ADDING to INDEX: {len(chunks)} chunks in batches of {max_batch_size}")
            
            for i in range(0, len(chunks), max_batch_size):
                batch_end = min(i + max_batch_size, len(chunks))
                batch_size = batch_end - i
                
                logger.warning(f"ADDING BATCH {i//max_batch_size + 1}/{(len(chunks) + max_batch_size - 1)//max_batch_size} to INDEX")
                log_memory_usage(f"before_add_batch_{i//max_batch_size + 1}")
                
                batch_ids = np.arange(start_id + i, start_id + batch_end, dtype=np.int64)
                batch_embeddings = embeddings[i:batch_end]
                
                self.index.add_with_ids(batch_embeddings, batch_ids)
                
                batch_ids = None
                batch_embeddings = None
                
                log_memory_usage(f"after_add_batch_{i//max_batch_size + 1}")
                gc.collect()
                log_memory_usage(f"after_gc_add_batch_{i//max_batch_size + 1}")
        else:
            logger.warning(f"ADDING {len(embeddings)} vectors to FAISS index")
            log_memory_usage("before_index_add")
            
            ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)
            self.index.add_with_ids(embeddings, ids)
            
            log_memory_usage("after_index_add")
        
        logger.warning(f"STORING {len(chunks)} chunks in memory")
        log_memory_usage("before_adding_chunks_to_list")
        
        for i in range(0, len(chunks), max_batch_size):
            batch_end = min(i + max_batch_size, len(chunks))
            self.chunks.extend(chunks[i:batch_end])
            gc.collect()
            
        log_memory_usage("after_adding_chunks_to_list")
        gc.collect()
        log_memory_usage("after_gc_chunks_to_list")
        
        logger.info(f"Added {len(chunks)} chunks to vector store. Total chunks: {len(self.chunks)}")
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[TextChunk, float]]:
        log_memory_usage("before_search")
        
        if not self.index or not self.chunks:
            logger.warning("Empty vector store or index not initialized")
            return []
            
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        if isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = 10
            
        logger.warning(f"SEARCHING INDEX: k={k}, total chunks={len(self.chunks)}")
        D, I = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx]
            results.append((chunk, float(score)))
            
        log_memory_usage("after_search")
        return results
    
    def save(self, directory: str):
        log_memory_usage("before_save")
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        if self.index:
            index_path = os.path.join(directory, "index.faiss")
            logger.warning(f"SAVING FAISS INDEX to {index_path}")
            faiss.write_index(self.index, index_path)
            logger.info(f"Saved Faiss index to {index_path}")
            
        chunks_data = []
        for chunk in self.chunks:
            chunk_data = {
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            chunks_data.append(chunk_data)
            
        chunks_path = os.path.join(directory, "chunks.json")
        logger.warning(f"SAVING {len(chunks_data)} CHUNKS to {chunks_path}")
        with open(chunks_path, "w") as f:
            json.dump(chunks_data, f)
        logger.info(f"Saved {len(chunks_data)} chunks to {chunks_path}")
        
        log_memory_usage("after_save")
            
    def load(self, directory: str):
        log_memory_usage("before_load")
        
        index_path = os.path.join(directory, "index.faiss")
        if os.path.exists(index_path):
            logger.warning(f"LOADING FAISS INDEX from {index_path}")
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded Faiss index from {index_path}")
        else:
            logger.error(f"Index file not found at {index_path}")
        
        chunks_path = os.path.join(directory, "chunks.json")
        if os.path.exists(chunks_path):
            logger.warning(f"LOADING CHUNKS from {chunks_path}")
            with open(chunks_path, "r") as f:
                chunks_data = json.load(f)
                
            self.chunks = []
            for chunk_data in chunks_data:
                chunk = TextChunk(chunk_data["text"], chunk_data["metadata"])
                self.chunks.append(chunk)
            logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_path}")
        else:
            logger.error(f"Chunks file not found at {chunks_path}")
            
        log_memory_usage("after_load")