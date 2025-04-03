import os
import gc
import numpy as np
from typing import List
from src.rag_components.ocr_processor import OCR
from src.rag_components.text_chunk import TextChunk
from src.rag_components.embedding import GeminiEmbeddingProvider
from src.rag_components.utils import log_memory_usage, log_array_info, logger, chunk_generator
from src.rag_components.config import DEFAULT_CONFIG, GOOGLE_API_KEY, MISTRAL_API_KEY


class DocumentProcessor:
    def __init__(self, ocr: OCR, embedding_provider: GeminiEmbeddingProvider, 
                 chunk_size: int = DEFAULT_CONFIG["chunk_size"], 
                 chunk_overlap: int = DEFAULT_CONFIG["chunk_overlap"]):
        self.ocr = ocr
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.warning(f"DocumentProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
    def process_pdf(self, pdf_path: str) -> List[TextChunk]:
        logger.warning(f"PROCESSING PDF: {pdf_path}")
        log_memory_usage("before_ocr")
        
        self.ocr.execute(pdf_path)
        
        log_memory_usage("after_ocr")
        
        text_by_page = self.ocr.get_text_by_page()
        logger.warning(f"OCR COMPLETE: Extracted {len(text_by_page)} pages of text")
        
        gc.collect()
        log_memory_usage("after_ocr_gc")
        
        token_info = self.ocr.count_tokens()
        
        all_chunks = []
        total_text_length = sum(len(text) for text in text_by_page.values())
        logger.warning(f"TOTAL TEXT LENGTH: {total_text_length} characters")
        
        for page_idx, page_text in text_by_page.items():
            logger.warning(f"CHUNKING PAGE {page_idx}: Text length {len(page_text)}")
            log_memory_usage(f"before_chunk_page_{page_idx}")
            
            page_chunks = self._chunk_text(page_text)
            
            logger.warning(f"CREATED {len(page_chunks)} CHUNKS FOR PAGE {page_idx}")
            
            for i, chunk in enumerate(page_chunks):
                metadata = {
                    "source": os.path.basename(pdf_path),
                    "page": page_idx,
                    "chunk_index": i,
                }
                
                all_chunks.append(TextChunk(chunk, metadata))
                
            page_chunks = None
            
            log_memory_usage(f"after_chunk_page_{page_idx}")
            gc.collect()
            log_memory_usage(f"after_gc_page_{page_idx}")
        
        text_by_page = None
        token_info = None
        
        logger.warning(f"CHUNKING COMPLETE: Created {len(all_chunks)} chunks")
        
        if len(all_chunks) > 100:
            logger.warning(f"LARGE NUMBER OF CHUNKS: {len(all_chunks)} - may cause memory issues")
            
        log_memory_usage("after_chunking")
        gc.collect()
        log_memory_usage("after_chunking_gc")
        return all_chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        logger.info("Starting _chunk_text")
        total_length = len(text)
        logger.info(f"Text length to chunk: {total_length}")
        if not text:
            return []
        
        chunks = []
        i = 0
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            logger.warning("chunk_overlap must be less than chunk_size. Using chunk_size as step.")
            step = self.chunk_size

        iteration = 0
        while i < total_length:
            if iteration % 10 == 0:
                log_memory_usage(f"_chunk_text iteration {iteration}")
            
            tentative_end = min(i + self.chunk_size, total_length)
            
            if tentative_end < total_length:
                paragraph_break = text.rfind("\n\n", i, tentative_end)
                if paragraph_break != -1 and paragraph_break > i + self.chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    sentence_break = text.rfind(". ", i, tentative_end)
                    if sentence_break != -1 and sentence_break > i + self.chunk_size // 2:
                        end = sentence_break + 2
                    else:
                        word_break = text.rfind(" ", i, tentative_end)
                        if word_break != -1 and word_break > i + self.chunk_size // 2:
                            end = word_break + 1
                        else:
                            end = tentative_end
            else:
                end = tentative_end
            
            chunk = text[i:end].strip()
            if chunk:
                chunks.append(chunk)
                if len(chunks) % 10 == 0:
                    logger.info(f"Created chunk {len(chunks)} with {len(chunk)} characters")
            
            i += step
            iteration += 1

        # Calculate and log chunk size statistics
        if chunks:
            chunk_sizes = [len(chunk) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunks)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            logger.info(f"_chunk_text completed: generated {len(chunks)} chunks")
            logger.info(f"Chunk size statistics: min={min_size}, avg={avg_size:.1f}, max={max_size} characters")
            
        log_memory_usage("_chunk_text completed")
        return chunks
    
    def embed_chunks(self, chunks: List[TextChunk]) -> np.ndarray:
        logger.warning(f"EMBEDDING {len(chunks)} CHUNKS")
        log_memory_usage("before_embed_chunks")
        
        max_batch_size = 10
        
        if len(chunks) > 50:
            logger.warning(f"HIGH MEMORY RISK: Very large batch of {len(chunks)} chunks")
            
        sample_size = min(len(chunks), 10)
        avg_text_len = sum(len(chunk.text) for chunk in chunks[:sample_size]) / sample_size
        est_total_len = int(avg_text_len * len(chunks))
        logger.warning(f"ESTIMATED TOTAL TEXT TO EMBED: ~{est_total_len} characters")
        
        if len(chunks) > max_batch_size:
            logger.warning(f"BATCH PROCESSING: Splitting {len(chunks)} chunks into batches of {max_batch_size}")
            
            dim = self.embedding_provider.embedding_dimension
            if dim:
                logger.warning(f"PRE-ALLOCATING embedding array: ({len(chunks)}, {dim})")
                embeddings = np.zeros((len(chunks), dim), dtype=np.float32)
                
                for i in range(0, len(chunks), max_batch_size):
                    batch_end = min(i + max_batch_size, len(chunks))
                    batch_chunks = chunks[i:batch_end]
                    batch_texts = [chunk.text for chunk in batch_chunks]
                    
                    logger.warning(f"PROCESSING BATCH {i//max_batch_size + 1}/{(len(chunks) + max_batch_size - 1)//max_batch_size}")
                    log_memory_usage(f"before_batch_{i//max_batch_size + 1}")
                    
                    batch_embeddings = self.embedding_provider.get_embeddings(batch_texts)
                    
                    embeddings[i:batch_end] = batch_embeddings
                    
                    batch_texts = None
                    batch_chunks = None
                    batch_embeddings = None
                    
                    log_memory_usage(f"after_batch_{i//max_batch_size + 1}")
                    gc.collect()
                    log_memory_usage(f"after_gc_batch_{i//max_batch_size + 1}")
            else:
                all_embeddings = []
                
                for i in range(0, len(chunks), max_batch_size):
                    batch_end = min(i + max_batch_size, len(chunks))
                    batch_chunks = chunks[i:batch_end]
                    batch_texts = [chunk.text for chunk in batch_chunks]
                    
                    logger.warning(f"PROCESSING BATCH {i//max_batch_size + 1}/{(len(chunks) + max_batch_size - 1)//max_batch_size}")
                    log_memory_usage(f"before_batch_{i//max_batch_size + 1}")
                    
                    batch_embeddings = self.embedding_provider.get_embeddings(batch_texts)
                    all_embeddings.append(batch_embeddings)
                    
                    if self.embedding_provider.embedding_dimension is None and batch_embeddings.shape[1]:
                        self.embedding_provider.embedding_dimension = batch_embeddings.shape[1]
                    
                    batch_texts = None
                    batch_chunks = None
                    
                    log_memory_usage(f"after_batch_{i//max_batch_size + 1}")
                    gc.collect()
                    log_memory_usage(f"after_gc_batch_{i//max_batch_size + 1}")
                
                logger.warning("CONCATENATING EMBEDDING BATCHES")
                log_memory_usage("before_concat_embeddings")
                embeddings = np.concatenate(all_embeddings, axis=0)
                all_embeddings = None
                log_memory_usage("after_concat_embeddings")
        else:
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_provider.get_embeddings(texts)
            texts = None
        
        log_array_info("embeddings_result", embeddings)
        
        # If embedding_dimension wasn't set before, set it now based on results
        if self.embedding_provider.embedding_dimension is None and embeddings.shape[1]:
            self.embedding_provider.embedding_dimension = embeddings.shape[1]
            logger.info(f"Setting embedding dimension to {embeddings.shape[1]} based on results")
        
        # Only check shape if we know the dimension
        if self.embedding_provider.embedding_dimension is not None:
            expected_shape = (len(chunks), self.embedding_provider.embedding_dimension)
            if embeddings.shape != expected_shape:
                logger.warning(f"UNEXPECTED EMBEDDING SHAPE: {embeddings.shape}, expected {expected_shape}")
        
        log_memory_usage("after_embed_chunks")
        gc.collect()
        log_memory_usage("after_embed_chunks_gc")
        return embeddings