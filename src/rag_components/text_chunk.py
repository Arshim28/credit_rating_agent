from typing import Dict, Any
from src.rag_components.utils import logger

class TextChunk:
	def __init__(self, text: str, metadata: Dict[str, Any]):
		self.text = text
		self.metadata = metadata
		logger.debug(f"Created TextChunk with length {len(text)} bytes")

	def __repr__(self):
		return f"TextChunk(len={len(self.text)}, metadata={self.metadata})"

	def __del__(self):
		try:
			self.text = None
			self.metadata = None
		except:
			pass

	