import pandas as pd
import json
import re
from typing import List, Dict
from datetime import datetime
import logging
import yaml
import spacy
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm
import os
import traceback

# Configure logging
logging.basicConfig(
    filename='speech_cleaner.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpeechCleaner:
    def __init__(self, config_path='/Users/fehmikatar/Desktop/final_project/cursor_project/src/patterns.yaml', chunk_size=400, overlap=50):
        # Load regex patterns from YAML configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.patterns = yaml.safe_load(f)
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_words = 50
        
        # Initialize spaCy for German
        self.nlp = spacy.load('de_core_news_sm')
        
        # Setup logging
        logging.basicConfig(filename='speech_cleaner.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    
    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks of ~400 words based on sentences."""
        try:
            print(f"\nCreating chunks from text of length: {len(text)}")
            
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            print(f"Found {len(sentences)} sentences")
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length > self.chunk_size:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        print(f"Created chunk of {len(chunk_text.split())} words")
                        chunks.append(chunk_text)
                    # Keep the last 'overlap' sentences for context
                    current_chunk = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else []
                    current_length = sum(len(s.split()) for s in current_chunk)
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add the last chunk if it exists
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                print(f"Created final chunk of {len(chunk_text.split())} words")
                chunks.append(chunk_text)
            
            # Filter out chunks that are too short
            valid_chunks = [chunk for chunk in chunks if len(chunk.split()) >= self.min_words]
            print(f"Created {len(valid_chunks)} valid chunks (min {self.min_words} words)")
            
            return valid_chunks
            
        except Exception as e:
            print(f"Error creating chunks: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning using regex and NLP."""
        if not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = re.sub(r'ß', 'ss', text)
        text = re.sub(r'ä', 'ae', text)
        text = re.sub(r'ö', 'oe', text)
        text = re.sub(r'ü', 'ue', text)
        
        # Apply all cleaning patterns with case-insensitive flag
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove everything up to and including the first exclamation mark
        text = re.sub(r'^[^!]*!', '', text)
        
        # Further cleaning steps
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\s,]+\.', '.', text)
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'^\s*[-–]\s*', '', text)
        text = re.sub(r'\s*[,;]\s*(?=[,;])', '', text)
        text = re.sub(r'!+', '!', text)
        text = re.sub(r'^[!.,\s]+', '', text)  # Remove leading punctuation
        
        return text.strip()
    
    def get_clean_speeches(self, csv_path: str, batch_size: int = 1000, year: int = 2000) -> List[Dict]:
        """Get cleaned speeches with chunking for long speeches."""
        try:
            print(f"Reading speeches from {csv_path}")
            
            # Read CSV file
            df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"Successfully read CSV with {len(df)} rows")
            
            # Ensure required columns exist
            required_columns = ['id', 'speechContent', 'firstName', 'lastName', 'factionId', 'date']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                print(f"Error: Missing columns in CSV: {missing_columns}")
                return []
            
            print(f"Found all required columns: {required_columns}")
            
            # Debug date values before conversion
            print("\nSample of date values before conversion:")
            print(df['date'].head())
            
            # Convert date column to datetime with error handling
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            print("\nSample of date values after conversion:")
            print(df['date'].head())
            
            # Check for NaT (Not a Time) values
            nat_count = df['date'].isna().sum()
            print(f"\nFound {nat_count} invalid dates")
            
            # Drop rows with invalid dates
            df = df.dropna(subset=['date'])
            print(f"After dropping invalid dates: {len(df)} rows")
            
            # Show year distribution
            year_counts = df['date'].dt.year.value_counts().sort_index()
            print("\nYear distribution:")
            print(year_counts)
            
            # Filter speeches from 2000 onwards
            df = df[df['date'].dt.year >= year]
            total_speeches = len(df)
            print(f"\nFound {total_speeches} speeches from {year} onwards")
            
            if total_speeches == 0:
                print("No speeches found after filtering!")
                return []
            
            # Process all speeches in batches with overall progress bar
            all_speeches = []
            total_batches = (total_speeches + batch_size - 1) // batch_size
            
            # Save progress every N batches
            save_interval = 5  # Save every 5 batches
            
            for batch_num in tqdm(range(total_batches), desc="Processing batches"):
                try:
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, total_speeches)
                    print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({start_idx} to {end_idx})")
                    
                    batch_df = df.iloc[start_idx:end_idx].copy()  # Make a copy to avoid memory leaks
                    
                    # Clean speeches in current batch
                    batch_speeches = []  # Store speeches for this batch
                    
                    for idx, row in batch_df.iterrows():
                        try:
                            original_text = row['speechContent']
                            print(f"Processing speech {idx} ({len(batch_speeches) + 1}/{batch_size})")
                            print(f"Original length: {len(original_text.split())} words")
                            
                            # Clean the text
                            cleaned_text = self.clean_text(original_text)
                            print(f"Cleaned length: {len(cleaned_text.split())} words")
                            print(f"Removed {len(original_text.split()) - len(cleaned_text.split())} words")
                            
                            # Only process if cleaned text is long enough
                            if len(cleaned_text.split()) > self.min_words:
                                # Create speech object
                                if len(cleaned_text.split()) > self.chunk_size:
                                    chunks = self.create_chunks(cleaned_text)
                                    for i, chunk in enumerate(chunks):
                                        speech = {
                                            'id': f"{row['id']}_chunk_{i+1}",
                                            'original_id': row['id'],
                                            'speaker': f"{row.get('firstName', '')} {row.get('lastName', '')}".strip(),
                                            'party': row.get('factionId', ''),
                                            'date': row['date'].strftime('%Y-%m-%d'),
                                            'original_text': original_text,
                                            'cleaned_text': chunk,
                                            'word_count': len(chunk.split()),
                                            'is_chunk': True,
                                            'chunk_number': i+1,
                                            'total_chunks': len(chunks)
                                        }
                                        batch_speeches.append(speech)
                                else:
                                    speech = {
                                        'id': row['id'],
                                        'original_id': row['id'],
                                        'speaker': f"{row.get('firstName', '')} {row.get('lastName', '')}".strip(),
                                        'party': row.get('factionId', ''),
                                        'date': row['date'].strftime('%Y-%m-%d'),
                                        'original_text': original_text,
                                        'cleaned_text': cleaned_text,
                                        'word_count': len(cleaned_text.split()),
                                        'is_chunk': False,
                                        'chunk_number': 1,
                                        'total_chunks': 1
                                    }
                                    batch_speeches.append(speech)
                        
                        except Exception as e:
                            print(f"Error processing speech {idx}: {e}")
                            continue
                    
                    # Add batch speeches to all speeches
                    all_speeches.extend(batch_speeches)
                    print(f"Processed {len(all_speeches)} total speeches so far")
                    
                    # Save progress every N batches
                    if (batch_num + 1) % save_interval == 0:
                        self._save_intermediate_results(all_speeches, batch_num + 1)
                    
                    # Clear batch data to free memory
                    del batch_df
                    del batch_speeches
                    
                except Exception as e:
                    print(f"Error processing batch {batch_num + 1}: {e}")
                    # Save what we have so far
                    self._save_intermediate_results(all_speeches, batch_num + 1)
                    continue
            
            return all_speeches
                
        except Exception as e:
            logging.error(f"Error processing speeches: {e}", exc_info=True)
            if all_speeches:  # Save what we have if there's an error
                self._save_intermediate_results(all_speeches, -1)  # -1 indicates error save
            return []
    
    def _save_intermediate_results(self, speeches: List[Dict], batch_num: int):
        """Save intermediate results to avoid losing progress."""
        try:
            output_path = f"/Users/fehmikatar/Desktop/final_project/intermediate_results_batch_{batch_num}.json"
            print(f"\nSaving intermediate results to: {output_path}")
            
            output_data = {
                'metadata': {
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'speeches_processed': len(speeches),
                    'batch_number': batch_num,
                    'description': 'Intermediate results'
                },
                'speeches': speeches
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved intermediate results for {len(speeches)} speeches")
            
        except Exception as e:
            print(f"Warning: Could not save intermediate results: {e}")
    
    def clean_speech(self, text: str, speech_id: str) -> List[Dict]:
        """Clean and chunk a speech text."""
        try:
            if not isinstance(text, str):
                print(f"Converting text type {type(text)} to string for speech {speech_id}")
                text = str(text)
            
            if not text.strip():
                print(f"Empty text for speech {speech_id}")
                return []
            
            print(f"\nProcessing speech {speech_id}")
            print(f"Original text length: {len(text)}")
            
            # Basic cleaning
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = ' '.join(text.split())
            
            # Remove procedural patterns
            for pattern in self.patterns['procedural']:
                text = re.sub(pattern, '', text)
            
            # Remove speaker references
            for pattern in self.patterns['speaker_refs']:
                text = re.sub(pattern, '', text)
            
            # Remove greetings
            for pattern in self.patterns['greetings']:
                text = re.sub(pattern, '', text)
            
            # Final cleanup
            text = ' '.join(text.split())
            text = text.strip()
            
            print(f"Cleaned text length: {len(text)}")
            
            if not text:
                print("Text was empty after cleaning")
                return []
            
            # Create chunks
            chunks = self.create_chunks(text)
            print(f"Created {len(chunks)} chunks")
            
            # Create chunk objects
            chunk_objects = []
            for i, chunk_text in enumerate(chunks, 1):
                word_count = len(chunk_text.split())
                print(f"Processing chunk {i}: {word_count} words")
                
                if word_count >= 50:
                    chunk_objects.append({
                        "id": f"{speech_id}_chunk_{i}",
                        "original_id": speech_id,
                        "text": chunk_text,
                        "word_count": word_count,
                        "is_chunk": True,
                        "chunk_number": i,
                        "total_chunks": len(chunks)
                    })
            
            print(f"Created {len(chunk_objects)} valid chunks")
            return chunk_objects
            
        except Exception as e:
            print(f"Error cleaning text for speech {speech_id}: {e}")
            print(f"Error details: {traceback.format_exc()}")
            return []
    
    def process_speeches_in_batches(self, speeches_df: pd.DataFrame, batch_size: int = 2000):
        """Process speeches in batches and save each batch."""
        try:
            total_speeches = len(speeches_df)
            print(f"\nTotal speeches to process: {total_speeches}")
            
            # Create output directory
            output_dir = "/Users/fehmikatar/Desktop/final_project/cursor_project/src/cleaned_json"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving cleaned batches to: {output_dir}")
            
            # Process in batches
            for batch_num, start_idx in enumerate(range(0, len(speeches_df), batch_size), 1):
                batch_df = speeches_df.iloc[start_idx:start_idx + batch_size]
                print(f"\nProcessing batch {batch_num}: speeches {start_idx} to {start_idx + len(batch_df)}")
                
                cleaned_speeches = []
                for _, speech in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_num}"):
                    try:
                        # Handle NaN values and type conversions
                        speech_id = str(int(speech['id'])) if pd.notna(speech['id']) else ''
                        speech_content = str(speech['speechContent']) if pd.notna(speech['speechContent']) else ''
                        first_name = str(speech['firstName']) if pd.notna(speech['firstName']) else ''
                        last_name = str(speech['lastName']) if pd.notna(speech['lastName']) else ''
                        speaker_name = f"{first_name} {last_name}".strip()
                        faction_id = str(int(speech['factionId'])) if pd.notna(speech['factionId']) else ''
                        
                        if not speech_content:
                            print(f"Warning: Empty speech content for ID {speech_id}")
                            continue
                        
                        chunks = self.clean_speech(speech_content, speech_id)
                        
                        if not chunks:
                            print(f"Warning: No valid chunks for speech {speech_id}")
                            continue
                        
                        for chunk in chunks:
                            cleaned_speeches.append({
                                'id': chunk['id'],
                                'original_id': speech_id,
                                'speaker': speaker_name,
                                'party': faction_id,
                                'date': str(speech.get('date', '')),
                                'text': chunk['text'],
                                'original_text': speech_content,
                                'word_count': chunk['word_count'],
                                'is_chunk': True,
                                'chunk_number': chunk['chunk_number'],
                                'total_chunks': chunk['total_chunks']
                            })
                    except Exception as e:
                        print(f"Error processing speech {speech.get('id')}: {e}")
                        print(traceback.format_exc())
                        continue
                
                # Save batch if we have cleaned speeches
                if cleaned_speeches:
                    output_file = os.path.join(output_dir, f"cleaned_text_{batch_num}.json")
                    print(f"\nSaving {len(cleaned_speeches)} cleaned speeches to {output_file}")
                    
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                'metadata': {
                                    'batch_number': batch_num,
                                    'speeches_in_batch': len(cleaned_speeches),
                                    'date_processed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                },
                                'speeches': cleaned_speeches
                            }, f, ensure_ascii=False, indent=2)
                        print(f"Successfully saved batch {batch_num}")
                    except Exception as e:
                        print(f"Error saving batch {batch_num}: {e}")
                else:
                    print(f"\nNo cleaned speeches in batch {batch_num}")
                
                # Clear memory
                del cleaned_speeches
                import gc
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            print(f"Error processing batch: {e}")
            raise

def main():
    try:
        csv_path = "/Users/fehmikatar/Desktop/speeches.csv"
        print(f"Loading speeches from: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"Error: speeches.csv not found at {csv_path}")
            print("Please make sure the file exists at the correct location")
            return
            
        # Load all speeches
        speeches_df = pd.read_csv(csv_path)
        print(f"Loaded total {len(speeches_df)} speeches")
        
        # Filter speeches after 23.11.2011
        speeches_df['date'] = pd.to_datetime(speeches_df['date'])
        cutoff_date = pd.to_datetime('2011-11-23')
        speeches_df = speeches_df[speeches_df['date'] > cutoff_date]
        print(f"Found {len(speeches_df)} speeches after 23.11.2011")
        
        # Print column names to debug
        print("\nColumns in CSV:", speeches_df.columns.tolist())
        
        # Update required columns to use factionId instead of party
        required_columns = ['id', 'speechContent', 'firstName', 'lastName', 'factionId', 'date']
        missing_columns = [col for col in required_columns if col not in speeches_df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return
        
        # Initialize cleaner
        cleaner = SpeechCleaner()
        
        # Process all filtered speeches in batches of 2000
        cleaner.process_speeches_in_batches(speeches_df, batch_size=2000)
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
