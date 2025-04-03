import os
import logging
import tiktoken
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from src.rag_components.config import MISTRAL_API_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    id: str
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    image_base64: str


@dataclass
class Dimensions:
    dpi: int
    height: int
    width: int


@dataclass
class Page:
    index: int
    markdown: str
    images: List[ImageData]
    dimensions: Dimensions


@dataclass
class UsageInfo:
    pages_processed: int
    doc_size_bytes: int


@dataclass
class OcrResponse:
    pages: List[Page]
    model: str
    usage_info: UsageInfo


@dataclass
class TokenInfo:
    total_tokens: int
    tokens_per_page: Dict[int, int]
    encoding_name: str


class OCR:
    def __init__(self, api_key: str = MISTRAL_API_KEY):
        self.api_key = api_key
        self.client = Mistral(api_key=self.api_key)
        self.pdf_id: Optional[str] = None
        self.pdf_url: Optional[str] = None
        self.pages: Optional[List[Page]] = None
        self.usage_info: Optional[UsageInfo] = None
        self.model: Optional[str] = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _upload_pdf(self, pdf_path: str) -> None:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        try:
            uploaded_pdf = self.client.files.upload(
                file={
                    "file_name": os.path.basename(pdf_path),
                    "content": open(pdf_path, "rb"),
                },
                purpose="ocr"
            )
            logger.info(f"Successfully uploaded PDF: {os.path.basename(pdf_path)}")
            
            self.pdf_id = uploaded_pdf.id
            signed_url_response = self.client.files.get_signed_url(file_id=self.pdf_id)
            self.pdf_url = signed_url_response.url  
        except Exception as e:
            logger.error(f"Error uploading PDF: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_ocr_response(self, pages=None, image_limit=None, image_min_size=None) -> OcrResponse:
        if not self.pdf_url:
            raise ValueError("No PDF URL found. Please upload a PDF first.")
            
        try:
            payload = {
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": self.pdf_url,
                },
                "include_image_base64": True
            }
            
            if pages is not None:
                payload["pages"] = pages
            
            if image_limit is not None:
                payload["image_limit"] = image_limit
                
            if image_min_size is not None:
                payload["image_min_size"] = image_min_size
            
            response = self.client.ocr.process(**payload)
            logger.info("Successfully processed document with OCR")
            
            self.pages = response.pages
            self.usage_info = response.usage_info
            self.model = response.model
            
            return OcrResponse(
                pages=self.pages,
                model=self.model,
                usage_info=self.usage_info
            )
        except Exception as e:
            logger.error(f"Error processing OCR: {e}")
            raise

    def execute(self, pdf_path: str, pages=None, image_limit=None, image_min_size=None) -> OcrResponse:
        self._upload_pdf(pdf_path)
        return self._get_ocr_response(pages=pages, image_limit=image_limit, image_min_size=image_min_size)

    def get_text(self) -> str:
        if not self.pages:
            raise ValueError("No OCR response found. Please run execute() method first.")
            
        return "".join(page.markdown for page in self.pages)
    
    def get_text_by_page(self) -> Dict[int, str]:
        """Returns a dictionary mapping page indices to their text content."""
        if not self.pages:
            raise ValueError("No OCR response found. Please run execute() method first.")
        
        return {page.index: page.markdown for page in self.pages}

    def get_images(self) -> List[ImageData]:
        if not self.pages:
            raise ValueError("No OCR response found. Please run execute() method first.")
            
        return [image for page in self.pages for image in page.images]

    def get_usage_info(self) -> UsageInfo:
        if not self.usage_info:
            raise ValueError("No OCR response found. Please run execute() method first.")
            
        return self.usage_info
    
    def count_tokens(self, encoding_name="cl100k_base") -> TokenInfo:
        """
        Count tokens in the extracted text using tiktoken.
        
        Args:
            encoding_name: The name of the encoding to use. 
                           Default is cl100k_base (used by GPT-3.5-turbo and GPT-4).
                           Other options include: p50k_base, r50k_base, etc.
        
        Returns:
            TokenInfo object with total token count and tokens per page
        """
        if not self.pages:
            raise ValueError("No OCR response found. Please run execute() method first.")
        
        try:
            # Initialize the tokenizer
            encoding = tiktoken.get_encoding(encoding_name)
            
            # Count tokens per page
            tokens_per_page = {}
            total_tokens = 0
            
            for page in self.pages:
                page_tokens = len(encoding.encode(page.markdown))
                tokens_per_page[page.index] = page_tokens
                total_tokens += page_tokens
            
            return TokenInfo(
                total_tokens=total_tokens,
                tokens_per_page=tokens_per_page,
                encoding_name=encoding_name
            )
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            raise


def analyze_for_embedding(text, encoding_name="cl100k_base") -> Tuple[int, Dict[str, int]]:
    """
    Analyze text for embedding compatibility with different services.
    
    Args:
        text: The text to analyze
        encoding_name: Encoding to use for tokenization
        
    Returns:
        Tuple of (token_count, compatibility_dict)
    """
    # Initialize the tokenizer
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = len(encoding.encode(text))
    
    # Common embedding service limits
    embedding_services = {
        "OpenAI Ada 002": 8191,
        "OpenAI text-embedding-3-small": 8191,
        "OpenAI text-embedding-3-large": 8191,
        "Cohere embed-english-v3.0": 512,
        "Cohere embed-multilingual-v3.0": 512,
        "Azure OpenAI Embeddings": 8191,
        "Mistral embed": 8192,
        "Vertex AI Embeddings": 3072,
        "Anthropic Embed": 9000,   # Approximate
        "Gemini embedding": 8000,  # As of March 2025
    }
    
    # Check compatibility
    compatibility = {}
    for service, limit in embedding_services.items():
        if tokens <= limit:
            compatibility[service] = "Compatible"
        else:
            compatibility[service] = f"Exceeds limit by {tokens - limit} tokens"
    
    return tokens, compatibility


if __name__ == "__main__":
    ocr = OCR()  # Using hardcoded API key
    
    # Process the document
    response = ocr.execute("test.pdf")
    
    # Get token information
    token_info = ocr.count_tokens()
    print(f"Total tokens in document: {token_info.total_tokens}")
    print(f"Tokens per page: {token_info.tokens_per_page}")
    
    # Get embedding compatibility
    text = ocr.get_text()
    token_count, compatibility = analyze_for_embedding(text)
    print("\nEmbedding Service Compatibility:")
    for service, status in compatibility.items():
        print(f"- {service}: {status}")
    
    # Print sample of extracted text
    print(f"\nText sample: {text[:100]}...")
    print(f"Number of images: {len(ocr.get_images())}")
    print(f"Usage info: {ocr.get_usage_info()}")