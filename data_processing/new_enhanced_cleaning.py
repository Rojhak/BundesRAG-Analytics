import json
import re
import logging
import spacy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import os
import traceback

# Configure logging for debugging and structured logging
logging.basicConfig(
    filename='speech_processor.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class EnhancedSpeechProcessor:
    def __init__(self, language: str = 'de'):
        """Initialize with minimal spaCy pipeline."""
        self.language = language
        print("Loading spaCy model...")
        try:
            # Use small model with minimal pipeline
            self.nlp = spacy.load(f"{self.language}_core_news_sm", 
                                 disable=["ner", "parser", "attribute_ruler", "lemmatizer"])
            self.nlp.add_pipe("sentencizer")
        except OSError:
            print(f"Downloading {self.language} language model...")
            spacy.cli.download(f"{self.language}_core_news_sm")
            self.nlp = spacy.load(f"{self.language}_core_news_sm",
                                 disable=["ner", "parser", "attribute_ruler", "lemmatizer"])
            self.nlp.add_pipe("sentencizer")

        self.version = "2.0"  # updated version

        # If you have a faction CSV, load it here
        self.faction_map = {}
        # self.load_factions()

        self.load_topic_taxonomy()
        self.load_unique_identifiers()
        self.initialize_additional_metadata()

    def load_factions(self):
        """Optional method to load faction/party info from CSV."""
        try:
            factions_path = Path("/Users/fehmikatar/Desktop/final_project/factions.csv")
            factions_df = pd.read_csv(factions_path)
            self.faction_map = dict(zip(factions_df['id'], factions_df['abbreviation']))
            logger.info(f"Loaded {len(self.faction_map)} factions.")
        except Exception as e:
            logger.error(f"Error loading factions: {e}")
            self.faction_map = {}

    def load_topic_taxonomy(self):
        """Load hierarchical topic taxonomy and assign IDs."""
        self.topic_taxonomy = {
            'Wirtschaft': {
                'keywords': [
                    'wirtschaft', 'arbeitsmarkt', 'unternehmen', 'industrie', 'handel',
                    'export', 'wachstum', 'konjunktur', 'steuer', 'finanzen', 'haushalt',
                    'investition', 'mittelstand', 'wettbewerb', 'förderung', 'markt', 'produktion'
                ],
                'subtopics': {
                    'Industrie': ['industrie', 'produktion', 'fertigung'],
                    'Finanzen': ['finanzen', 'steuer', 'haushalt'],
                    'Handel': ['handel', 'export', 'markt']
                }
            },
            'Migration': {
                'keywords': [
                    'migration', 'flüchtling', 'asyl', 'integration', 'einwanderung',
                    'ausländer', 'zuwanderung', 'aufnahme', 'unterbringung', 'geflüchtete'
                ],
                'subtopics': {
                    'Asyl': ['asyl', 'aufnahme'],
                    'Integration': ['integration', 'zuwanderung']
                }
            },
            'Umwelt': {
                'keywords': [
                    'umwelt', 'klima', 'naturschutz', 'nachhaltigkeit', 'erneuerbare',
                    'energie', 'emission', 'recycling', 'öko', 'klimaschutz', 'atomkraft'
                ],
                'subtopics': {
                    'Klimaschutz': ['klimaschutz', 'klima'],
                    'Nachhaltigkeit': ['nachhaltigkeit', 'erneuerbare']
                }
            },
            'Bildung': {
                'keywords': [
                    'bildung', 'schule', 'universität', 'ausbildung', 'forschung',
                    'wissenschaft', 'studium', 'lehre', 'hochschule', 'student', 'bildungspolitik'
                ],
                'subtopics': {
                    'Schulbildung': ['schule', 'bildungspolitik'],
                    'Hochschulbildung': ['universität', 'hochschule', 'studium', 'forschung']
                }
            },
            'Soziales': {
                'keywords': [
                    'sozial', 'rente', 'gesundheit', 'pflege', 'familie', 'kinder',
                    'jugend', 'alter', 'armut', 'versicherung', 'sozialpolitik'
                ],
                'subtopics': {
                    'Gesundheit': ['gesundheit', 'pflege', 'versicherung'],
                    'Sozialpolitik': ['sozialpolitik', 'rente', 'armut']
                }
            },
            'Sicherheit': {
                'keywords': [
                    'sicherheit', 'polizei', 'kriminalität', 'verteidigung', 'bundeswehr',
                    'terror', 'gewalt', 'innere sicherheit', 'cyber', 'kriminalitätsbekämpfung'
                ],
                'subtopics': {
                    'Innere Sicherheit': ['innere sicherheit', 'polizei'],
                    'Verteidigung': ['verteidigung', 'bundeswehr']
                }
            },
            'Außenpolitik': {
                'keywords': [
                    'außenpolitik', 'international', 'eu', 'europa', 'nato', 'vereinte nationen',
                    'diplomatie', 'ausland', 'beziehungen', 'zusammenarbeit', 'partnerschaft'
                ],
                'subtopics': {
                    'Diplomatie': ['diplomatie', 'diplomatische'],
                    'Internationale Beziehungen': ['internationale', 'beziehungen', 'europa']
                }
            },
            'Digitalisierung': {
                'keywords': [
                    'digital', 'internet', 'technologie', 'innovation', 'künstliche intelligenz',
                    'ki', 'online', 'digitalisierung', 'plattform', 'netzwerk', 'cyber'
                ],
                'subtopics': {
                    'Künstliche Intelligenz': ['künstliche intelligenz', 'ki'],
                    'Cybersecurity': ['cyber', 'sicherheit']
                }
            },
            'Infrastruktur': {
                'keywords': [
                    'infrastruktur', 'verkehr', 'bahn', 'straßen', 'mobilität', 'transport',
                    'breitband', 'bau', 'wohnen', 'städte', 'kommunen'
                ],
                'subtopics': {
                    'Verkehr': ['verkehr', 'bahn', 'straßen', 'mobilität', 'transport'],
                    'Bau und Wohnen': ['bau', 'wohnen', 'städte', 'kommunen']
                }
            },
            'Demokratie': {
                'keywords': [
                    'demokratie', 'rechtsstaat', 'verfassung', 'grundrechte', 'parlament',
                    'bürgerbeteiligung', 'wahlen', 'opposition', 'regierung', 'parteien'
                ],
                'subtopics': {
                    'Rechtsstaat': ['rechtsstaat', 'verfassung', 'grundrechte'],
                    'Politische Strukturen': ['parlament', 'regierung', 'parteien']
                }
            },
            'Gesundheit': {
                'keywords': [
                    'gesundheit', 'medizin', 'krankenhaus', 'pflege', 'therapie',
                    'krankenversicherung', 'gesundheitssystem', 'krankenkasse', 'apotheke'
                ],
                'subtopics': {
                    'Krankenhauswesen': ['krankenhaus', 'pflege'],
                    'Gesundheitssystem': ['gesundheitssystem', 'krankenversicherung', 'krankenkasse']
                }
            },
            'Technologie': {
                'keywords': [
                    'technologie', 'innovation', 'forschung', 'entwicklung', 'software',
                    'hardware', 'robotik', 'biotechnologie', 'informationstechnologie'
                ],
                'subtopics': {
                    'Informationstechnologie': ['informationstechnologie', 'software', 'hardware'],
                    'Robotik und Biotechnologie': ['robotik', 'biotechnologie']
                }
            }
        }

        # Assign unique IDs
        self.topic_id_map = {}
        topic_counter = 1
        subtopic_counter = 1
        for topic, details in self.topic_taxonomy.items():
            topic_id = f"T{topic_counter:03d}"
            self.topic_id_map[topic] = topic_id
            topic_counter += 1
            for subtopic in details['subtopics']:
                subtopic_id = f"ST{subtopic_counter:03d}"
                self.topic_id_map[subtopic] = subtopic_id
                subtopic_counter += 1

        # For numeric references
        self.number_id_map = {}
        self.number_counter = 1

        # Topic weights
        self.topic_weights = {
            'Wirtschaft': 1.2,
            'Migration': 1.2,
            'Umwelt': 1.2,
            'Bildung': 1.2,
            'Außenpolitik': 1.2,
            'Digitalisierung': 1.2,
            'Infrastruktur': 1.2,
            'Soziales': 0.8,
            'Demokratie': 0.8,
            'Gesundheit': 1.0,
            'Technologie': 1.0
        }

    def load_unique_identifiers(self):
        """Placeholder, already done in `load_topic_taxonomy` above."""
        pass

    def initialize_additional_metadata(self):
        """Placeholder for future enhancements."""
        pass

    def clean_text(self, text: str) -> str:
        """Basic cleaning."""
        if not isinstance(text, str):
            return ""

        text = ' '.join(text.split())
        text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
        text = re.sub(r'[^\w\s.!?]', ' ', text)
        return text

    def detect_topics_improved(self, text: str, threshold: float = 0.3) -> Dict[str, float]:
        """Enhanced topic detection with better handling of migration and legal topics."""
        try:
            text = str(text).lower()
            topic_scores = {}
            important_words = set(word.lower() for word in text.split())
            
            # Add legal reference detection
            legal_refs = re.findall(r'\bart\.\s?\d+[a-z]?\b|\b§\s?\d+[a-z]?\b|\basylrecht\b|\basylgesetz\b', text.lower())
            if legal_refs:
                logger.debug(f"Found legal references: {legal_refs}")
                topic_scores['Migration'] = max(topic_scores.get('Migration', 0), 0.8)
                topic_scores['Recht'] = max(topic_scores.get('Recht', 0), 0.8)

            for topic, details in self.topic_taxonomy.items():
                # Main topic keywords
                matches = sum(1 for w in important_words if w in [k.lower() for k in details['keywords']])
                weighted = matches * 0.6  # Main topics get higher weight
                
                # Subtopic handling with separate weighting
                for subtopic, sub_keywords in details['subtopics'].items():
                    sub_matches = sum(1 for w in important_words if w in [k.lower() for k in sub_keywords])
                    sub_weighted = sub_matches * 0.4  # Subtopics get lower weight
                    
                    if sub_weighted > threshold:
                        topic_scores[subtopic] = round(sub_weighted, 2)
                        # Boost main topic score when subtopics are found
                        weighted += sub_weighted * 0.3
                
                if weighted > threshold:
                    topic_scores[topic] = round(weighted, 2)
                    logger.debug(f"Topic {topic} detected with score {weighted:.2f}")
                    logger.debug(f"Matching keywords: {[w for w in important_words if w in [k.lower() for k in details['keywords']]]}")

            # Enhanced fallback for migration-specific terms
            migration_terms = {
                'flüchtling', 'asyl', 'migration', 'einwanderung', 'integration',
                'ausländer', 'zuwanderung', 'abschiebung', 'aufenthaltsrecht',
                'asylbewerber', 'migrant', 'aufnahme'
            }
            
            migration_matches = sum(1 for term in migration_terms if term in text)
            if migration_matches > 0:
                topic_scores['Migration'] = max(topic_scores.get('Migration', 0), 
                                             round(migration_matches * 0.4, 2))

            return topic_scores

        except Exception as e:
            logger.error(f"Error in topic detection: {e}")
            return {}

    def extract_number_context_improved(self, text: str) -> List[Dict]:
        """Extract numbers with contexts."""
        patterns = [
            r'\b\d+(?:\.\d+)?(?:,\d+)?\s*(?:Millionen|Million|Mio\.?|Milliarden|Mrd\.?|Euro|€|DM)\b',
            r'\b(?:[1-9]\d|100)(?:,\d+)?\s*(?:Prozent|%)\b',
            r'\b(?:19|20)\d{2}\b',
            r'\b\d{4,}(?:\.\d{3})*(?:,\d+)?\b'
        ]

        text = str(text)  # ensure fresh buffer
        doc = self.nlp(text)
        sents = list(doc.sents)
        contexts = []

        for pat in patterns:
            regex = re.compile(pat)
            for match in regex.finditer(text):
                start_pos = match.start()
                containing_sent = None
                for sent in sents:
                    if sent.start_char <= start_pos <= sent.end_char:
                        containing_sent = sent
                        break
                if containing_sent:
                    cstart = max(0, containing_sent.start_char - 30)
                    cend = min(len(text), containing_sent.end_char + 30)
                    snippet = text[cstart:cend].strip()

                    # find related topics
                    related_topics = set()
                    for tname, det in self.topic_taxonomy.items():
                        # check main keywords
                        if any(kw.lower() in snippet.lower() for kw in det['keywords']):
                            related_topics.add(tname)
                        # subtopics
                        for st in det['subtopics']:
                            if any(kw.lower() in snippet.lower() for kw in det['subtopics'][st]):
                                related_topics.add(st)

                    contexts.append({
                        'number': match.group(),
                        'contexts': [snippet],
                        'topics': list(related_topics)
                    })
        return self.deduplicate_number_contexts(contexts)

    def deduplicate_number_contexts(self, contexts: List[Dict]) -> List[Dict]:
        unique = {}
        for ctx in contexts:
            nm = ctx['number']
            if nm not in unique:
                unique[nm] = {
                    'id': f"N{self.number_counter:05d}",
                    'number': nm,
                    'contexts': set(),
                    'frequency': 0,
                    'topics': set()
                }
                self.number_counter += 1

            unique[nm]['contexts'].update(ctx['contexts'])
            unique[nm]['frequency'] += 1
            unique[nm]['topics'].update(ctx.get('topics', []))

        out = []
        for dat in unique.values():
            out.append({
                'id': dat['id'],
                'number': dat['number'],
                'contexts': list(dat['contexts'])[:2],
                'frequency': dat['frequency'],
                'topics': list(dat['topics'])
            })
        return out

    def infer_topics(self, text: str) -> List[str]:
        """Enhanced fallback topic inference with better prioritization."""
        try:
            text = str(text).lower()
            doc = self.nlp(text)
            
            # Get important nouns and their frequencies
            nouns = [t.lemma_.lower() for t in doc if t.pos_ == 'NOUN' and not t.is_stop]
            if not nouns:
                return []
            
            freq = pd.Series(nouns).value_counts()
            top_nouns = freq.head(5).index.tolist()  # Increased from 3 to 5
            
            out = []
            for noun in top_nouns:
                # Check main topics
                for topic, details in self.topic_taxonomy.items():
                    if noun in [k.lower() for k in details['keywords']]:
                        out.append(topic)
                        continue
                    
                    # Check subtopics with higher priority
                    for st, keywords in details['subtopics'].items():
                        if noun in [k.lower() for k in keywords]:
                            out.append(st)
                            out.append(topic)  # Also add parent topic
            
            # Deduplicate while preserving order
            return list(dict.fromkeys(out))
            
        except Exception as e:
            logger.error(f"Error in topic inference: {e}")
            return []

    def prepare_speech_as_chunk(self, speech_data: Dict) -> Dict:
        """Process each speech as a single chunk."""
        try:
            # Get text from the correct field - it's in 'text' not 'cleaned_text'
            text = speech_data.get('text', '')
            if not text:
                print(f"Warning: No text found for speech {speech_data.get('id', '')}")
                return {}
            
            print(f"\nProcessing speech {speech_data.get('id', '')}")
            print(f"Original text length: {len(text)}")
            
            # Clean the text
            cleaned_text = self.clean_text(text)
            print(f"Cleaned text length: {len(cleaned_text)}")
            
            # detect topics & numbers
            topics = self.detect_topics_improved(cleaned_text)
            numbers = self.extract_number_context_improved(cleaned_text)
            
            print(f"Found topics: {list(topics.keys())}")
            print(f"Found {len(numbers)} numbers")

            topic_list = [
                {'id': self.topic_id_map.get(t, f'T{hash(t) % 10000:04d}'), 
                 'name': t, 
                 'confidence': sc}
                for t, sc in sorted(topics.items(), key=lambda x: x[1], reverse=True)
            ]

            if topic_list:
                avg_conf = sum(t['confidence'] for t in topic_list)/len(topic_list)
            else:
                avg_conf = 0
            
            num_topics = len(topic_list)
            num_numbers = sum(n['frequency'] for n in numbers)
            retriever_priority = (avg_conf*0.6 + (num_topics/10)*0.3 + (num_numbers/5)*0.1)
            retriever_priority = min(round(retriever_priority, 2), 1.0)

            # Create summary
            summary_parts = []
            if topic_list:
                main_tops = topic_list[:2]
                topic_str = " and ".join(m['name'] for m in main_tops)
                summary_parts.append(f"Discussion of {topic_str}")
            else:
                # fallback to inferred
                guessed = self.infer_topics(cleaned_text)
                if guessed:
                    topic_str = " and ".join(guessed[:2])
                    summary_parts.append(f"Discussion of {topic_str}")

            if numbers:
                knums = [n['number'] for n in numbers[:2]]
                nstr = f"mentioning {' and '.join(knums)}"
                summary_parts.append(nstr)

            summary = " ".join(summary_parts)+'.' if summary_parts else "No significant topics or numbers discussed."

            retrievable_text = {
                "summary": summary,
                "key_topics": [
                    {"id": t['id'], "name": t['name'], "confidence": t['confidence']}
                    for t in topic_list[:3]
                ],
                "key_numbers": [
                    {"id": n['id'], "number": n['number'], "frequency": n['frequency']}
                    for n in numbers[:2]
                ],
                "context": " ".join(c for n in numbers[:2] for c in n['contexts']) if numbers else "No contextual information."
            }

            chunk = {
                'chunk_id': str(speech_data.get('id', '')),
                'speech_id': str(speech_data.get('original_id', speech_data.get('id', ''))),
                'chunk_number': 1,
                'is_chunk': True,
                'retrievable_text': retrievable_text,
                'retriever_priority': retriever_priority,
                'cleaned_text': cleaned_text,  # Now we're setting the cleaned text
                'summary': summary,
                'topics': topic_list,
                'numbers': numbers,
                'word_count': len(cleaned_text.split()) if cleaned_text else 0
            }
            return chunk
            
        except Exception as e:
            print(f"Error preparing chunk: {e}")
            print(traceback.format_exc())
            return {}

    def aggregate_topics(self, chunks: List[Dict]) -> List[Dict]:
        """Aggregate topics across what is effectively '1 chunk per speech'."""
        topic_data = {}
        for c in chunks:
            for tp in c['topics']:
                tn = tp['name']
                if tn not in topic_data:
                    topic_data[tn] = {'mentions': 0, 'confidence_sum': 0.0}
                topic_data[tn]['mentions'] += 1
                topic_data[tn]['confidence_sum'] += tp['confidence']

        out = []
        for tn, dat in sorted(topic_data.items(), key=lambda x: (x[1]['mentions'], x[1]['confidence_sum']), reverse=True):
            out.append({
                'id': self.topic_id_map.get(tn, ''),
                'name': tn,
                'mentions': dat['mentions'],
                'avg_confidence': round(dat['confidence_sum']/dat['mentions'], 2) if dat['mentions'] else 0
            })
        return out

    def aggregate_numbers(self, chunks: List[Dict]) -> List[Dict]:
        """Aggregate numbers."""
        number_data = {}
        for c in chunks:
            for n in c['numbers']:
                if n['number'] not in number_data:
                    number_data[n['number']] = {
                        'id': n['id'],
                        'number': n['number'],
                        'contexts': set(),
                        'frequency': 0,
                        'topics': set()
                    }
                nd = number_data[n['number']]
                nd['contexts'].update(n['contexts'])
                nd['frequency'] += n['frequency']
                nd['topics'].update(n.get('topics', []))

        agg = []
        for num, dat in sorted(number_data.items(), key=lambda x: x[1]['frequency'], reverse=True):
            agg.append({
                'id': dat['id'],
                'number': dat['number'],
                'contexts': list(dat['contexts'])[:2],
                'frequency': dat['frequency'],
                'topics': list(dat['topics'])
            })
        return agg

    def generate_global_retrievable_summary(self, speech: Dict) -> str:
        """Speech-level summary text."""
        meta = speech.get('metadata', {})
        tops = sorted(speech.get('global_topics', []), key=lambda x: (x['mentions'], x['avg_confidence']), reverse=True)[:3]
        top_str = ", ".join(f"{t['name']} ({t['mentions']} mentions, confidence: {t['avg_confidence']:.2f})" for t in tops)

        if speech.get('global_numbers'):
            top_nums = speech['global_numbers'][:2]
            num_str = " Key numbers: " + ", ".join(f"{n['number']} ({n['frequency']} mentions)" for n in top_nums) + "."
        else:
            num_str = " Key numbers: No key numbers."

        if num_str and not num_str.endswith('.'):
            num_str += '.'

        return (
            f"Speech by {meta.get('speaker', '')} ({meta.get('party', '')}) on {meta.get('date', '')}. "
            f"Global topics: {top_str}.{num_str} "
            f"Total chunks: {meta.get('total_chunks', 1)}."
        )

    def format_speech_summary(self, speech: Dict) -> str:
        """Pretty print for debugging."""
        m = speech.get('metadata', {})
        lines = [
            f"Speech ID: {speech.get('id', '')}",
            f"Speaker: {m.get('speaker', '')} ({m.get('party', '')})",
            f"Date: {m.get('date', '')}",
            f"Total Words: {m.get('total_words', 0)}",
            f"Chunks: {m.get('total_chunks', 1)}",
            "\nGlobal Topics:",
            "--------------"
        ]

        for top in speech.get('global_topics', [])[:5]:
            lines.append(
                f"- {top['name']} (ID: {top['id']}): {top['mentions']} mentions "
                f"(avg. confidence: {top['avg_confidence']:.2f})"
            )

        if speech.get('global_numbers'):
            lines.append("\nKey Numbers:")
            lines.append("------------")
            for num in speech['global_numbers'][:3]:
                lines.append(f"\n{num['number']} (ID: {num['id']}) - mentioned {num['frequency']} times")
                for c in num['contexts'][:2]:
                    lines.append(f"  Context: \"{c}\"")

        lines.append("\nChunk Summaries:")
        lines.append("---------------")

        for ch in speech.get('chunks', []):
            lines.extend([
                f"\nChunk {ch['chunk_id']}:",
                f"Word count: {ch['word_count']}",
                f"Summary: {ch['summary']}"
            ])

        return '\n'.join(lines)

    def convert_sets_to_lists(self, obj):
        """Recursively convert all sets to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self.convert_sets_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_sets_to_lists(i) for i in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj

    def validate_output(self, processed_speeches: List[Dict]):
        """Validate for warning logs, but won't raise exceptions."""
        for sp in processed_speeches:
            chunks = sp.get('chunks', [])
            for c in chunks:
                if not c.get('cleaned_text'):
                    logger.warning(f"Chunk {c.get('chunk_id')} has empty cleaned_text.")

                awc = len(c.get('cleaned_text', '').split())
                if awc != c.get('word_count', 0):
                    logger.warning(
                        f"Chunk {c.get('chunk_id')} word count mismatch. "
                        f"Expected {c.get('word_count')}, got {awc}."
                    )

                rt = c.get('retrievable_text', {})
                if not rt:
                    logger.warning(f"Chunk {c.get('chunk_id')} has empty retrievable_text.")
                else:
                    sm = rt.get('summary', '')
                    if not sm.strip():
                        logger.warning(f"Chunk {c.get('chunk_id')} has empty summary in retrievable_text.")

                    kt = rt.get('key_topics', [])
                    kn = rt.get('key_numbers', [])
                    cx = rt.get('context', '')

                    if not isinstance(kt, list):
                        logger.warning(f"Chunk {c.get('chunk_id')} has invalid key_topics format.")
                    if not isinstance(kn, list):
                        logger.warning(f"Chunk {c.get('chunk_id')} has invalid key_numbers format.")
                    if not isinstance(cx, str) or not cx.strip():
                        logger.warning(f"Chunk {c.get('chunk_id')} has empty context in retrievable_text.")

                if not c.get('topics'):
                    logger.warning(f"Chunk {c.get('chunk_id')} has no detected topics.")

                for num in c.get('numbers', []):
                    if not num.get('contexts'):
                        logger.warning(f"Number {num.get('number')} in chunk {c.get('chunk_id')} has no contexts.")

            if not sp.get('global_topics'):
                logger.warning(f"Speech {sp.get('id')} has no global topics.")
            if not sp.get('global_numbers'):
                logger.warning(f"Speech {sp.get('id')} has no global numbers.")

    def analyze_cross_speech_patterns(self, speeches: List[Dict]) -> Dict:
        """Analyze patterns across multiple speeches (still works the same)."""
        analysis = {
            'topic_frequency': {},
            'topic_confidence': {},
            'party_topics': {},
            'temporal_patterns': {},
            'topic_relationships': {}
        }

        for sp in speeches:
            meta = sp.get('metadata', {})
            gtops = sp.get('global_topics', [])
            party = meta.get('party', '')
            date = meta.get('date', '')
            year = date.split('-')[0] if date else 'unknown'

            for t in gtops:
                tn = t['name']
                analysis['topic_frequency'][tn] = analysis['topic_frequency'].get(tn, 0) + t['mentions']
                if tn not in analysis['topic_confidence']:
                    analysis['topic_confidence'][tn] = []
                analysis['topic_confidence'][tn].append(t['avg_confidence'])

                if party not in analysis['party_topics']:
                    analysis['party_topics'][party] = {}
                if tn not in analysis['party_topics'][party]:
                    analysis['party_topics'][party][tn] = 0
                analysis['party_topics'][party][tn] += t['mentions']

                if year not in analysis['temporal_patterns']:
                    analysis['temporal_patterns'][year] = {}
                if tn not in analysis['temporal_patterns'][year]:
                    analysis['temporal_patterns'][year][tn] = 0
                analysis['temporal_patterns'][year][tn] += t['mentions']

            sp_names = [t['name'] for t in gtops]
            for i in range(len(sp_names)):
                for j in range(i+1, len(sp_names)):
                    pair = tuple(sorted([sp_names[i], sp_names[j]]))
                    analysis['topic_relationships'][pair] = analysis['topic_relationships'].get(pair, 0) + 1

        for tn in analysis['topic_confidence']:
            scs = analysis['topic_confidence'][tn]
            if scs:
                analysis['topic_confidence'][tn] = round(sum(scs)/len(scs), 2)
            else:
                analysis['topic_confidence'][tn] = 0

        rels = []
        for pair, freq in analysis['topic_relationships'].items():
            rels.append({
                'topics': list(pair),
                'frequency': freq
            })
        analysis['topic_relationships'] = rels

        return analysis

    def generate_global_summary(self, speeches: List[Dict], cross_speech_analysis: Dict) -> str:
        """Comprehensive summary text across all speeches."""
        total_speeches = len(speeches)
        total_words = sum(sp.get('metadata', {}).get('total_words', 0) for sp in speeches)

        lines = [
            "Global Analysis Summary",
            "=" * 50,
            f"Total Speeches Analyzed: {total_speeches}",
            f"Total Words Processed: {total_words}",
            "\nTop Topics Across All Speeches:",
            "-" * 30
        ]

        freq_map = cross_speech_analysis.get('topic_frequency', {})
        conf_map = cross_speech_analysis.get('topic_confidence', {})
        sorted_topics = sorted(freq_map.items(), key=lambda x: (x[1], conf_map.get(x[0], 0)), reverse=True)
        for topic, freq in sorted_topics[:8]:
            cval = conf_map.get(topic, 0)
            lines.append(f"- {topic}: mentioned {freq} times (avg. confidence: {cval:.2f})")

        lines.append("\nParty Focus Areas:")
        lines.append("-"*20)
        pmap = cross_speech_analysis.get('party_topics', {})
        for prt, tmap in pmap.items():
            top3 = sorted(tmap.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(f"\n{prt}:")
            for tn, ccount in top3:
                lines.append(f"- {tn}: {ccount} mentions")

        lines.append("\nTemporal Patterns:")
        lines.append("-"*20)
        tmap = cross_speech_analysis.get('temporal_patterns', {})
        for yr in sorted(tmap):
            items = tmap[yr]
            top3 = sorted(items.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(f"\n{yr}:")
            for tp, ct in top3:
                lines.append(f"- {tp}: {ct} mentions")

        lines.append("\nInter-Topic Relationships:")
        lines.append("-"*30)
        rel_list = cross_speech_analysis.get('topic_relationships', [])
        for rel in rel_list:
            topics_ = " & ".join(rel['topics'])
            fr = rel['frequency']
            lines.append(f"- {topics_}: co-mentioned {fr} times")

        return "\n".join(lines)

    def process_speeches(self, input_path: str, sample_size: Optional[int] = None) -> List[Dict]:
        """Process speeches in batches of 2000."""
        try:
            logger.info(f"Loading speeches from {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_speeches = data.get('speeches', [])
                
            total_speeches = len(all_speeches)
            print(f"Found {total_speeches} total speeches")
            
            # Create output directory if it doesn't exist
            output_dir = "/Users/fehmikatar/Desktop/final_project/processed_batches"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Will save batches to: {output_dir}")
            
            processed = []
            batch_size = 2000
            
            # Process in batches
            for start_idx in range(0, total_speeches, batch_size):
                end_idx = min(start_idx + batch_size, total_speeches)
                print(f"\nProcessing batch: speeches {start_idx} to {end_idx}")
                
                batch_speeches = all_speeches[start_idx:end_idx]
                batch_processed = []
                
                for speech in tqdm(batch_speeches, desc="Processing speeches"):
                    try:
                        # Your existing processing logic
                        big_chunk = self.prepare_speech_as_chunk(speech)
                        rag_chunks = [big_chunk]
                        gtopics = self.aggregate_topics(rag_chunks)
                        gnumbers = self.aggregate_numbers(rag_chunks)
                        
                        sp_summary = {
                            'id': str(speech.get('id', '')),
                            'metadata': {
                                'speaker': str(speech.get('speaker', '')),
                                'party': str(self.faction_map.get(speech.get('party'), speech.get('party', ''))),
                                'date': str(speech.get('date', '')),
                                'total_chunks': 1,
                                'total_words': big_chunk['word_count'],
                                'original_text': speech.get('original_text', ''),
                                'duration_minutes': speech.get('duration_minutes', 0),
                                'speech_type': speech.get('speech_type', 'general'),
                                'language': self.language
                            },
                            'chunks': rag_chunks,
                            'global_topics': gtopics,
                            'global_numbers': gnumbers,
                            'global_retrievable_summary': self.generate_global_retrievable_summary({
                                'metadata': {
                                    'speaker': str(speech.get('speaker', '')),
                                    'party': str(self.faction_map.get(speech.get('party'), speech.get('party', ''))),
                                    'date': str(speech.get('date', '')),
                                    'total_chunks': 1
                                },
                                'global_topics': gtopics,
                                'global_numbers': gnumbers
                            })
                        }
                        batch_processed.append(sp_summary)
                    except Exception as ex:
                        logger.exception(f"Error processing speech {speech.get('id')}: {ex}")
                
                # Save this batch
                if batch_processed:
                    batch_output = {
                        'version': self.version,
                        'metadata': {
                            'batch_number': start_idx // batch_size + 1,
                            'speeches_in_batch': len(batch_processed),
                            'start_index': start_idx,
                            'end_index': end_idx
                        },
                        'speeches': batch_processed
                    }
                    
                    batch_filename = f"processed_speeches_batch_{start_idx // batch_size + 1}.json"
                    batch_path = os.path.join(output_dir, batch_filename)
                    
                    print(f"\nSaving batch to: {batch_path}")
                    with open(batch_path, 'w', encoding='utf-8') as f:
                        json.dump(batch_output, f, ensure_ascii=False, indent=2)
                
                # Add batch to processed list
                processed.extend(batch_processed)
                
                # Clear memory
                del batch_processed
                import gc
                gc.collect()
                
            return processed
            
        except Exception as e:
            logger.exception(f"Error loading speeches: {e}")
            return []

    def main(self, input_path: str, output_path: str, sample_size: Optional[int] = None):
        """ Main driver: read JSON, process each speech as single chunk, analyze, save. """
        speeches = self.process_speeches(input_path, sample_size=sample_size)
        if speeches:
            self.validate_output(speeches)
            cross_speech_analysis = self.analyze_cross_speech_patterns(speeches)
            global_summary = self.generate_global_summary(speeches, cross_speech_analysis)

            out_data = {
                'version': self.version,
                'metadata': {
                    'original_speeches': len(speeches),
                    'total_chunks': sum(len(sp['chunks']) for sp in speeches),
                    'description': 'Enhanced analysis (no re-chunking) with hierarchical taxonomy'
                },
                'global_analysis': cross_speech_analysis,
                'global_summary': global_summary,
                'speeches': speeches
            }

            out_data = self.convert_sets_to_lists(out_data)

            print(f"\nSaving processed speeches to {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)

            print("\nDone! Global Summary:")
            print(global_summary)
        else:
            print("No speeches were processed.")

    def test_topic_detection(self, sample_size: int = 20):
        """Test topic detection on a small sample of speeches."""
        try:
            print(f"\nTesting topic detection on {sample_size} speeches...")
            
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_speeches = data.get('speeches', [])[:sample_size]
            
            results = []
            for speech in tqdm(test_speeches, desc="Testing speeches"):
                text = speech.get('text', '')
                
                # Test main topic detection
                topics = self.detect_topics_improved(text)
                
                # Test fallback if no topics found
                if not topics:
                    fallback_topics = self.infer_topics(text)
                else:
                    fallback_topics = []
                
                results.append({
                    'id': speech.get('id', ''),
                    'detected_topics': topics,
                    'fallback_topics': fallback_topics,
                    'text_preview': text[:200] + '...' if len(text) > 200 else text
                })
            
            # Save test results
            output_path = "topic_detection_test.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'test_size': sample_size,
                    'results': results
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\nTest results saved to: {output_path}")
            
            # Print summary
            print("\nTopic Detection Summary:")
            print(f"Total speeches tested: {len(results)}")
            print(f"Speeches with detected topics: {sum(1 for r in results if r['detected_topics'])}")
            print(f"Speeches using fallback: {sum(1 for r in results if r['fallback_topics'])}")
            
        except Exception as e:
            print(f"Error in topic detection test: {e}")
            logger.error(traceback.format_exc())

#################
# Usage Example #
#################

def run_no_chunk():
    processor = EnhancedSpeechProcessor(language='de')
    input_path = "/Users/fehmikatar/Desktop/final_project/intermediate_results_batch_160.json"   # Adjust
    output_path = "test_new_enhanced_cleaning.json" # Adjust
    processor.main(input_path, output_path, sample_size=None)

if __name__ == "__main__":
    run_no_chunk()
