# retriever.py
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import torch
from creat_summary import SummaryProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | retriever | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler()  # Only log to console for now
    ]
)
logger = logging.getLogger(__name__)

class Retriever:
    """
    Retriever class to handle semantic search in a FAISS index of speech chunks.
    """
    def __init__(self, vector_store_path: str = None):
        """Initialize the retriever with paths to FAISS index and metadata."""
        # Set environment variables for better resource management
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        
        if vector_store_path is None:
            vector_store_path = os.path.join(os.path.dirname(__file__), 'vector_store')
        
        self.index_path = os.path.join(vector_store_path, 'speeches.index')
        self.metadata_path = os.path.join(vector_store_path, 'chunk_metadata.json')
        
        # Initialize model with minimal settings
        self.model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            device='cpu'
        )
        self.model.max_seq_length = 384  # Limit sequence length
        
        self.load_index()
        self.load_metadata()
        self.summarizer = SummaryProcessor(language="de")
        logger.info("Retriever initialization complete")

    def __del__(self):
        """Cleanup resources when the object is destroyed"""
        try:
            # Clean up FAISS resources
            if hasattr(self, 'index'):
                del self.index
            
            # Clean up model resources
            if hasattr(self, 'model'):
                del self.model
            
            # Clean up other large objects
            if hasattr(self, 'chunk_metadata'):
                del self.chunk_metadata
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    def load_index(self) -> bool:
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded FAISS index from {self.index_path}")

            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.chunk_metadata = json.load(f)
                logger.info(f"Loaded {len(self.chunk_metadata)} metadata entries")

            logger.info(f"Index loaded with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error loading index or metadata: {e}", exc_info=True)
            return False

    def load_metadata(self) -> bool:
        """Load chunk metadata from JSON file."""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.chunk_metadata = json.load(f)
            logger.info(f"Loaded {len(self.chunk_metadata)} metadata entries")
            return True
        except Exception as e:
            logger.error(f"Error loading metadata: {e}", exc_info=True)
            self.chunk_metadata = []
            return False

    def get_embedding(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embedding.reshape(1, -1).astype('float32')

    def get_context_snippet(self, text: str, window: int = None) -> str:
        """Return the full text without truncation"""
        return text

    def _apply_filters(self, result: dict, filters: dict) -> bool:
        if not filters:
            return True

        metadata = result.get('metadata', {})

        # party filter
        party_filter = filters.get('party')
        if party_filter and metadata.get('party', '') != party_filter:
            return False

        # year range filter
        year_range = filters.get('year_range')
        if year_range:
            date_str = metadata.get('date', '')
            if date_str:
                try:
                    year = int(date_str.split('-')[0])
                    start_year, end_year = year_range
                    if not (start_year <= year <= end_year):
                        return False
                except (ValueError, IndexError):
                    return False

        # topic filter
        topic_filter = filters.get('topic')
        if topic_filter:
            topics = result.get('topics', [])
            topic_names = [t.get('name', t) if isinstance(t, dict) else t for t in topics]
            if topic_filter not in topic_names:
                return False

        return True

    def search(self, query: str, top_k: int = 5, filters: dict = None) -> List[Dict]:
        """Search for relevant chunks."""
        try:
            query_vector = self.get_embedding(query)
            D, I = self.index.search(query_vector, top_k * 2)
            
            results = []
            for i, idx in enumerate(I[0]):
                if idx < 0 or idx >= len(self.chunk_metadata):
                    continue
                    
                chunk = self.chunk_metadata[idx]
                
                # Get the complete text
                cleaned_text = chunk.get('cleaned_text', '')
                original_text = chunk.get('metadata', {}).get('original_text', '')
                
                # Get context from retrievable_text
                context = chunk.get('retrievable_text', {}).get('context', '')
                if not context:
                    # Fallback to metadata context
                    context = chunk.get('metadata', {}).get('context', '')
                
                # Generate a new summary using the summarizer
                generated_summary = self.summarizer.generate_summary(cleaned_text)
                
                # Get top 3 topics only
                topics = chunk.get('topics', [])[:3] if chunk.get('topics') else []
                
                # Create result with generated summary and original context
                result = {
                    'chunk_id': chunk.get('chunk_id', ''),
                    'speech_id': chunk.get('speech_id', ''),
                    'summary': generated_summary,  # Use the newly generated summary
                    'cleaned_text': cleaned_text,
                    'context': context,  # Use original context from JSON
                    'metadata': chunk.get('metadata', {}),
                    'topics': topics,  # Limited to top 3 topics
                    'original_text': original_text,
                    'score': float(D[0][i])
                }
                
                # Apply filters
                if not self._apply_filters(result, filters):
                    continue
                    
                results.append(result)
                
                if len(results) >= top_k:
                    break
                    
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error in search: {e}", exc_info=True)
            return []