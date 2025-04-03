import gc
import tracemalloc
from typing import List, Dict, Any, Optional
from src.rag_components.ocr_processor import OCR
from src.rag_components.embedding import GeminiEmbeddingProvider
from src.rag_components.document_processor import DocumentProcessor
from src.rag_components.vector_store import FaissVectorStore
from src.rag_components.utils import log_memory_usage, logger
from src.rag_components.config import GOOGLE_API_KEY, MISTRAL_API_KEY, DEFAULT_CONFIG

class OCRVectorStore:
    """
    Combines OCR processing, embeddings, and vector storage to enable
    semantic search on PDF documents.
    """
    
    def __init__(self, index_type: str = DEFAULT_CONFIG["index_type"], 
                 chunk_size: int = DEFAULT_CONFIG["chunk_size"], 
                 chunk_overlap: int = DEFAULT_CONFIG["chunk_overlap"]):
        """
        Initialize the OCR vector store.
        
        Args:
            index_type: Type of FAISS index to use (Flat, HNSW, IVF, IVFPQ)
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
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
        
    def add_document(self, pdf_path: str) -> bool:
        """
        Process a PDF document and add it to the vector store.
        
        Args:
            pdf_path: Path to the PDF document
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.warning(f"ADDING DOCUMENT: {pdf_path}")
        log_memory_usage("before_add_document")
        
        tracemalloc.start()
        max_chunks_per_batch = 30  # Smaller batches for less memory pressure
        
        try:
            # Step 1: Process PDF to extract text chunks
            chunks = self.processor.process_pdf(pdf_path)
            
            if not chunks:
                logger.warning(f"No text extracted from {pdf_path}")
                tracemalloc.stop()
                return False
                
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
            
            current, peak = tracemalloc.get_traced_memory()
            logger.warning(f"MEMORY AFTER PDF PROCESSING: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
            
            # Step 2: Process chunks in batches to manage memory
            total_batches = (len(chunks) + max_chunks_per_batch - 1) // max_chunks_per_batch
            
            for i in range(0, len(chunks), max_chunks_per_batch):
                batch_end = min(i + max_chunks_per_batch, len(chunks))
                batch_chunks = chunks[i:batch_end]
                
                batch_num = i // max_chunks_per_batch + 1
                logger.warning(f"PROCESSING BATCH {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
                
                # Step 2a: Generate embeddings for this batch
                logger.warning(f"GENERATING EMBEDDINGS FOR BATCH {batch_num}")
                batch_embeddings = self.processor.embed_chunks(batch_chunks)
                
                current, peak = tracemalloc.get_traced_memory()
                logger.warning(f"MEMORY AFTER BATCH EMBEDDING: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
                
                # Step 2b: Add embeddings to vector store
                logger.warning(f"ADDING BATCH {batch_num} TO VECTOR STORE")
                self.vector_store.add_chunks(batch_chunks, batch_embeddings)
                
                current, peak = tracemalloc.get_traced_memory()
                logger.warning(f"MEMORY AFTER ADDING BATCH TO VECTOR STORE: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
                
                # Clean up batch variables to free memory
                batch_chunks = None
                batch_embeddings = None
                gc.collect()
                
                current, peak = tracemalloc.get_traced_memory()
                logger.warning(f"MEMORY AFTER BATCH GC: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
                
            tracemalloc.stop()
            # Clean up main variables
            chunks = None
            log_memory_usage("after_add_document")
            gc.collect()
            log_memory_usage("after_gc")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {str(e)}")
            tracemalloc.stop()
            gc.collect()
            return False
        
    def answer_question(self, question: str, k: int = 30) -> List[Dict[str, Any]]:
        """
        Search the vector store for chunks relevant to a question.
        
        Args:
            question: The question to answer
            k: Number of results to return
            
        Returns:
            List of dictionaries with text, metadata, and relevance score
        """
        logger.warning(f"ANSWERING QUESTION: {question}")
        log_memory_usage("before_question")
        
        logger.warning("GENERATING QUESTION EMBEDDING")
        question_embedding = self.embedding_provider.get_embeddings([question])[0]
        
        logger.warning(f"SEARCHING VECTOR STORE WITH K={k}")
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
    
    def save(self, directory: str) -> bool:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save the vector store
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Saving vector store to {directory}")
        log_memory_usage("before_save")
        
        try:
            self.vector_store.save(directory)
            log_memory_usage("after_save")
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
        
    def load(self, directory: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            directory: Directory to load the vector store from
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Loading vector store from {directory}")
        log_memory_usage("before_load")
        
        try:
            self.vector_store.load(directory)
            log_memory_usage("after_load")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False