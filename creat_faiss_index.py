# creat_faiss_index.py

import faiss
import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import argparse
import gc
import torch
import sys
from typing import List, Dict, Optional
import re

def setup_logging():
    log_dir = os.path.expanduser('~/Desktop/logs/')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | creat_faiss_index | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'index_creator.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_party_mapping(csv_path: str) -> Dict[str, str]:
    """
    Load party mapping (faction IDs to party names) from a CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
        return dict(zip(df['id'], df['party']))
    except Exception as e:
        logger.error(f"Error loading party mapping: {e}")
        return {}

def process_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 4) -> Optional[np.ndarray]:
    """
    Encode a list of texts in small batches, returning a 2D numpy array of embeddings.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_embeddings = model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            continue
    
    if embeddings:
        return np.vstack(embeddings)
    return None

def clean_original_text(text: str) -> str:
    """
    Clean original text by removing parentheses and their contents.
    
    Args:
        text (str): Original text to clean
        
    Returns:
        str: Cleaned text with parentheses and their contents removed
    """
    # Remove parentheses and their contents
    cleaned = re.sub(r'\([^)]*\)', '', text)
    # Remove multiple spaces and clean up
    cleaned = ' '.join(cleaned.split())
    return cleaned

def process_all_json_files(base_dir: str, max_batch: int = 63) -> List[dict]:
    """
    Process JSON files up to a specified batch number.
    
    Args:
        base_dir (str): Base directory containing processed_batches
        max_batch (int): Maximum batch number to process
        
    Returns:
        List[dict]: Combined list of speeches
    """
    all_speeches = []
    processed_dir = os.path.join(base_dir, 'processed_batches')
    
    try:
        # Get and sort files by batch number
        files = []
        for filename in os.listdir(processed_dir):
            if filename.endswith('.json'):
                try:
                    batch_num = int(filename.split('_')[-1].split('.')[0])
                    if batch_num <= max_batch:
                        files.append((batch_num, filename))
                except ValueError:
                    continue
        
        # Sort by batch number
        files.sort(key=lambda x: x[0])
        
        # Process files
        for _, filename in files:
            file_path = os.path.join(processed_dir, filename)
            logger.info(f"Processing {filename}...")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    speeches = data.get('speeches', [])
                    all_speeches.extend(speeches)
                    logger.info(f"Added {len(speeches)} speeches from {filename}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing {filename}: {str(e)}")
                logger.info(f"Skipping corrupted file: {filename}")
                continue
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
                
        logger.info(f"Total speeches collected: {len(all_speeches)}")
        return all_speeches
        
    except Exception as e:
        logger.error(f"Error processing JSON files: {e}")
        return []

def create_faiss_index(
    base_dir: str,
    vector_store_path: str,
    factions_csv_path: Optional[str] = None,
    embedding_model: str = 'sentence-transformers/all-mpnet-base-v2',
    dimension: int = 768,
    batch_size: int = 4
) -> int:
    """
    Create FAISS index from all processed speeches.
    """
    try:
        # Setup environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        # Initialize counters and storage
        processed_speeches = 0
        total_vectors = 0
        all_metadata = []

        # Initialize FAISS index
        logger.info("Initializing FAISS index...")
        index = faiss.IndexFlatL2(dimension)

        # Load model
        logger.info("Loading model...")
        model = SentenceTransformer(embedding_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        model.max_seq_length = 384

        # Load party mapping
        party_mapping = {'-1': 'Unknown Party'}
        if factions_csv_path:
            try:
                df = pd.read_csv(factions_csv_path)
                party_mapping.update(dict(zip(df['id'].astype(str), df['full_name'])))
            except Exception as e:
                logger.error(f"Could not load faction CSV: {e}")

        # Process all JSON files
        speeches = process_all_json_files(base_dir)
        total_speeches = len(speeches)
        logger.info(f"Will process {total_speeches} speeches")

        # Process in batches
        local_batch_size = min(batch_size, 10)
        for batch_idx in range(0, len(speeches), local_batch_size):
            try:
                batch_speeches = speeches[batch_idx:batch_idx + local_batch_size]
                
                for speech in batch_speeches:
                    try:
                        speech_id = speech.get('id', '')  # Get the parent speech ID
                        metadata = speech.get('metadata', {})
                        
                        # Get speech-level metadata
                        party_id = str(metadata.get('party', '-1'))
                        party_name = party_mapping.get(party_id, "Unknown Party")
                        speaker = metadata.get('speaker', 'Unknown Speaker')
                        date = metadata.get('date', 'Unknown')
                        speech_type = metadata.get('speech_type', 'general')
                        
                        # Clean original text by removing parentheses and their contents
                        original_text = metadata.get('original_text', '')
                        cleaned_original_text = clean_original_text(original_text)

                        chunks = speech.get('chunks', [])
                        for chunk_idx, chunk in enumerate(chunks):
                            try:
                                text = chunk.get('cleaned_text', '').strip()
                                if len(text) < 10:
                                    continue

                                # Ensure chunk has proper ID format
                                chunk_id = f"{speech_id}_chunk_{chunk_idx + 1}"
                                
                                # Combine chunk-level topics with global speech-level topics
                                chunk_topics = chunk.get('topics', [])
                                speech_topics = speech.get('global_topics', [])
                                combined_topics = chunk_topics + [
                                    t for t in speech_topics
                                    if t not in chunk_topics
                                ]

                                # Create chunk metadata with proper speech reference
                                chunk_metadata = {
                                    'party': party_name,
                                    'party_id': party_id,
                                    'speaker_party': party_name,
                                    'speech_type': speech_type,
                                    'speaker': speaker,
                                    'date': date,
                                    'speech_id': speech_id,  # Store the parent speech ID
                                    'chunk_id': chunk_id,
                                    'chunk_number': chunk.get('chunk_number', chunk_idx + 1),
                                    'topics': combined_topics,
                                    'retrievable_text': chunk.get('retrievable_text', {}),
                                    'context': chunk.get('contexts', ''),
                                    'word_count': len(text.split()),
                                    'language': metadata.get('language', 'de'),
                                    'global_retrievable_summary': speech.get('global_retrievable_summary', ''),
                                    'original_text': cleaned_original_text  # Use cleaned original text
                                }

                                # Generate embedding
                                try:
                                    with torch.no_grad():
                                        embedding = model.encode(
                                            text,
                                            convert_to_numpy=True,
                                            normalize_embeddings=True,
                                            show_progress_bar=False
                                        )
                                    embedding = embedding.astype('float32').reshape(1, -1)
                                    index.add(embedding)

                                    # Store metadata with proper structure
                                    all_metadata.append({
                                        'chunk_id': chunk_id,
                                        'speech_id': speech_id,  # Include speech_id in top level
                                        'cleaned_text': text[:1000],
                                        'metadata': chunk_metadata,
                                        'topics': combined_topics,
                                        'retrievable_text': chunk.get('retrievable_text', {}),
                                        'context': chunk.get('contexts', ''),
                                        'summary': chunk.get('summary', '')
                                    })
                                    total_vectors += 1

                                except Exception as e:
                                    logger.error(f"Error encoding chunk {chunk_id}: {str(e)}")
                                    continue

                            except Exception as e:
                                logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                                continue

                        processed_speeches += 1
                        if processed_speeches % 5 == 0:
                            logger.info(f"Processed {processed_speeches}/{total_speeches} speeches")

                    except Exception as e:
                        logger.error(f"Error processing speech: {str(e)}")
                        continue

                if total_vectors > 0 and processed_speeches % 10 == 0:
                    logger.info(
                        f"Progress: {processed_speeches}/{total_speeches} speeches, {total_vectors} vectors"
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if total_vectors > 0:
            os.makedirs(vector_store_path, exist_ok=True)
            faiss.write_index(index, os.path.join(vector_store_path, 'speeches.index'))
            
            with open(os.path.join(vector_store_path, 'chunk_metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Index saved with {total_vectors} vectors from {processed_speeches} speeches")
            return total_vectors

    except Exception as e:
        logger.error(f"Fatal error in index creation: {e}", exc_info=True)
        return 0

def main():
    parser = argparse.ArgumentParser(description="Create FAISS index from all processed speeches.")
    parser.add_argument('--base_dir', type=str, required=True,
                       help="Base directory containing processed_batches folder")
    parser.add_argument('--vector_store', type=str, required=True,
                       help="Path to save vector store")
    parser.add_argument('--factions_csv', type=str, default=None,
                       help="Path to factions CSV file (optional)")
    parser.add_argument('--batch_size', type=int, default=4,
                       help="Batch size for processing")

    args = parser.parse_args()
    
    create_faiss_index(
        base_dir=args.base_dir,
        vector_store_path=args.vector_store,
        factions_csv_path=args.factions_csv,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()