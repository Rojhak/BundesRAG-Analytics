# creat_summary.py
import json
import logging
import spacy
from typing import Dict, List, Optional
from tqdm import tqdm
import re
import os
from transformers import pipeline

# Create logs directory if it doesn't exist
log_dir = '/Users/fehmikatar/Desktop/final_project/rag_model/logs'
os.makedirs(log_dir, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | summary_processor | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'summary_processor.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SummaryProcessor:
    """
    SummaryProcessor class to generate extractive summaries for speeches.
    Uses a simple scoring approach (sentence length, position, keywords).
    """
    def __init__(self, language: str = 'de'):
        """
        Initialize the SummaryProcessor with spaCy.

        Args:
            language (str, optional): Language code for spaCy. Defaults to 'de'.
        """
        self.language = language
        self.nlp = spacy.load(f'{language}_core_news_sm')
        logger.info("Initialized basic summarizer without model")

    def clean_text(self, text: str) -> str:
        """
        Basic cleaning of text (whitespace, new lines).

        Args:
            text (str): The text to clean.

        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str):
            return ""
        text = ' '.join(text.split())
        return text

    def generate_summary(self, text: str, max_length: int = 500, min_length: int = 100) -> str:
        """
        Generate a summary of the given text.
        Args:
            text: Text to summarize
            max_length: Maximum length of summary (increased from 130 to 500)
            min_length: Minimum length of summary (increased from 30 to 100)
        """
        try:
            # Process the text
            doc = self.nlp(text)
            
            # Get all sentences
            sentences = [sent.text.strip() for sent in doc.sents]
            
            # If text is short enough, return as is
            if len(' '.join(sentences)) <= max_length:
                return ' '.join(sentences)
            
            # Score sentences based on position and length
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                score = 0
                # Position score - earlier sentences get higher scores
                score += 1.0 / (i + 1)
                
                # Length score - prefer medium length sentences
                length = len(sentence.split())
                if 10 <= length <= 50:  # Increased upper limit
                    score += 0.3
                
                scored_sentences.append((score, sentence))
            
            # Sort by score and select top sentences
            scored_sentences.sort(reverse=True)
            
            # Take sentences until we reach max_length
            summary_sentences = []
            current_length = 0
            
            for _, sentence in scored_sentences:
                if current_length + len(sentence) > max_length:
                    break
                summary_sentences.append(sentence)
                current_length += len(sentence)
                
                if current_length >= min_length and len(summary_sentences) >= 3:  # Ensure at least 3 sentences
                    break
            
            # Join sentences and return
            return ' '.join(summary_sentences)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

    def generate_global_summary(self, speech: Dict) -> str:
        """
        Generate a global summary for a speech based on metadata.

        Args:
            speech (dict): The speech dictionary, which contains 'metadata',
                           'global_topics', and 'global_numbers'.

        Returns:
            str: A text summary that includes speaker, party, date, top topics, and numbers.
        """
        meta = speech.get('metadata', {})
        global_topics = speech.get('global_topics', [])
        global_numbers = speech.get('global_numbers', [])

        speaker = meta.get('speaker', 'Unknown Speaker')
        party = meta.get('party', 'Unknown Party')
        date = meta.get('date', 'Unknown Date')

        # Format top 3 topics
        topic_str = ", ".join([
            f"{t['name']} ({t['mentions']} mentions, confidence: {t['avg_confidence']:.2f})"
            for t in global_topics[:3]
        ])
        # Format top 2 numbers
        if global_numbers:
            numbers_str = ", ".join([
                f"{n['number']} ({n['frequency']} mentions)" 
                for n in global_numbers[:2]
            ])
        else:
            numbers_str = "No key numbers."

        summary = (
            f"Speech by {speaker} ({party}) on {date}. "
            f"Global topics: {topic_str}. "
            f"Key numbers: {numbers_str}. "
            f"Total chunks: {meta.get('total_chunks', 1)}."
        )
        logger.debug(f"Generated global summary: {summary[:100]}...")
        return summary

    def process_summaries(self, input_path: str, output_path: str):
        """
        Load speeches from input JSON, regenerate summaries, and save them to output path.

        Args:
            input_path (str): Path to the input JSON file with speeches.
            output_path (str): Path to save the updated JSON file.
        """
        try:
            logger.info(f"Loading speeches from {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                speeches = data.get('speeches', [])

            for speech in tqdm(speeches, desc="Generating Summaries"):
                try:
                    # Aggregate all cleaned_text from chunks
                    all_chunks = speech.get('chunks', [])
                    full_text = ' '.join([chunk.get('cleaned_text', '') for chunk in all_chunks])
                    full_text = self.clean_text(full_text)

                    if not full_text:
                        logger.warning(f"Speech {speech.get('id')} has no text.")
                        speech['global_retrievable_summary'] = self.generate_global_summary(speech)
                        continue

                    summary = self.generate_summary(full_text, max_length=130, min_length=30)
                    logger.info(f"Generated summary for speech {speech.get('id')}: {summary[:100]}...")

                    # Update global retrievable summary
                    speech['global_retrievable_summary'] = summary

                except Exception as ex:
                    logger.exception(f"Error processing speech {speech.get('id')}", exc_info=True)

            # Save updated speeches
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Updated summaries saved to {output_path}")

            print(f"Summaries successfully regenerated and saved to {output_path}")

        except Exception as e:
            logger.exception(f"Error in processing summaries: {e}")
            print(f"An error occurred: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarize and update speeches JSON file.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input JSON file with speeches.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the updated JSON file.")

    args = parser.parse_args()

    processor = SummaryProcessor(language='de')
    processor.process_summaries(args.input, args.output)

if __name__ == "__main__":
    main()