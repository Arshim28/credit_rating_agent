import gc
import tracemalloc
from typing import List, Dict
from ocr_processor import OCR
from embedding import GeminiEmbeddingProvider
from document_processor import DocumentProcessor
from vector_store import FaissVectorStore
from utils import log_memory_usage, logger, chunk_generator
from config import GOOGLE_API_KEY, MISTRAL_API_KEY, DEFAULT_CONFIG

class OCRVectorStore:
    def __init__(self, index_type: str = DEFAULT_CONFIG["index_type"], 
                 chunk_size: int = DEFAULT_CONFIG["chunk_size"], 
                 chunk_overlap: int = DEFAULT_CONFIG["chunk_overlap"]):
        logger.warning("INITIALIZING OCR VECTOR STORE")
        log_memory_usage("init_start")
        
        self.ocr = OCR(api_key=MISTRAL_API_KEY)
        
        self.embedding_provider = GeminiEmbeddingProvider(
            api_key=GOOGLE_API_KEY,
            dimension=DEFAULT_CONFIG["embedding_dimension"]
        )
        
        # Log the actual values being used
        logger.info(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        self.processor = DocumentProcessor(
            ocr=self.ocr,
            embedding_provider=self.embedding_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Don't initialize the vector store with a dimension yet
        # We'll initialize it properly after getting the first embeddings
        self.vector_store = FaissVectorStore(
            dimension=None,  # Will be set when first embeddings are created
            index_type=index_type
        )
        
        log_memory_usage("init_complete")
        
    def add_document(self, pdf_path: str):
        logger.warning(f"ADDING DOCUMENT: {pdf_path}")
        log_memory_usage("before_add_document")
        
        tracemalloc.start()
        
        max_chunks_per_batch = 50
        
        chunks = self.processor.process_pdf(pdf_path)
        
        if not chunks:
            logger.warning(f"No text extracted from {pdf_path}")
            tracemalloc.stop()
            return
            
        logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
        
        current, peak = tracemalloc.get_traced_memory()
        logger.warning(f"MEMORY AFTER PDF PROCESSING: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
        
        if len(chunks) > max_chunks_per_batch:
            logger.warning(f"PROCESSING LARGE DOCUMENT IN BATCHES: {len(chunks)} chunks in batches of {max_chunks_per_batch}")
            
            for i in range(0, len(chunks), max_chunks_per_batch):
                batch_end = min(i + max_chunks_per_batch, len(chunks))
                batch_chunks = chunks[i:batch_end]
                
                logger.warning(f"PROCESSING BATCH {i//max_chunks_per_batch + 1}/{(len(chunks) + max_chunks_per_batch - 1)//max_chunks_per_batch}")
                
                logger.warning(f"GENERATING EMBEDDINGS FOR BATCH ({len(batch_chunks)} chunks)")
                batch_embeddings = self.processor.embed_chunks(batch_chunks)
                
                current, peak = tracemalloc.get_traced_memory()
                logger.warning(f"MEMORY AFTER BATCH EMBEDDING: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
                
                logger.warning(f"ADDING BATCH TO VECTOR STORE")
                self.vector_store.add_chunks(batch_chunks, batch_embeddings)
                
                current, peak = tracemalloc.get_traced_memory()
                logger.warning(f"MEMORY AFTER ADDING BATCH TO VECTOR STORE: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
                
                batch_chunks = None
                batch_embeddings = None
                gc.collect()
                
                current, peak = tracemalloc.get_traced_memory()
                logger.warning(f"MEMORY AFTER BATCH GC: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
        else:
            logger.warning("GENERATING EMBEDDINGS")
            embeddings = self.processor.embed_chunks(chunks)
            
            current, peak = tracemalloc.get_traced_memory()
            logger.warning(f"MEMORY AFTER EMBEDDING: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
            
            logger.warning("ADDING TO VECTOR STORE")
            self.vector_store.add_chunks(chunks, embeddings)
            
            current, peak = tracemalloc.get_traced_memory()
            logger.warning(f"MEMORY AFTER ADDING TO VECTOR STORE: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
            
            embeddings = None
        
        tracemalloc.stop()
        
        chunks = None
        log_memory_usage("after_add_document")
        gc.collect()
        log_memory_usage("after_gc")
        
    def answer_question(self, question: str, k: int = 50) -> List[Dict]:
        logger.warning(f"ANSWERING QUESTION: {question}")
        log_memory_usage("before_question")
        
        logger.warning("GENERATING QUESTION EMBEDDING")
        question_embedding = self.embedding_provider.get_embeddings([question])[0]
        
        logger.warning("SEARCHING VECTOR STORE")
        results = self.vector_store.search(question_embedding, k=k)
        
        formatted_results = []
        for chunk, score in results:
            formatted_results.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": score
            })
            
        log_memory_usage("after_question")
        return formatted_results
    
    def save(self, directory: str):
        logger.info(f"Saving vector store to {directory}")
        log_memory_usage("before_save")
        self.vector_store.save(directory)
        log_memory_usage("after_save")
        
    def load(self, directory: str):
        logger.info(f"Loading vector store from {directory}")
        log_memory_usage("before_load")
        self.vector_store.load(directory)
        log_memory_usage("after_load")