# semantic_chunker.py
import nltk
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import logging

# Import MAX_CHUNK_TOKENS from config to ensure consistency for RecursiveCharacterTextSplitter if it's used directly
from .config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Ensure required NLTK data (canonical IDs)
def _ensure_nltk_resource(find_path: str, download_name: str):
    try:
        nltk.data.find(find_path)
    except LookupError:
        nltk.download(download_name)

_ensure_nltk_resource('tokenizers/punkt', 'punkt')
_ensure_nltk_resource('chunkers/maxent_ne_chunker', 'maxent_ne_chunker')
_ensure_nltk_resource('corpora/words', 'words')
_ensure_nltk_resource('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')

class SemanticChunker:
    """
    Semantic chunking for medical documents using NLTK and langchain.
    Preserves document structure and semantic boundaries.
    """
    def __init__(self):
        self.sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        # Initialize RecursiveCharacterTextSplitter with values from config for consistency
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_TOKENS, # Use MAX_CHUNK_TOKENS from config
            chunk_overlap=CHUNK_OVERLAP, # Use CHUNK_OVERLAP from config
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        sections = []
        section_patterns = [
            r'^\s*\d+\.?\s+([A-Z\s]+)$',  # Numbered sections
            r'^([A-Z][A-Z\s]{3,})$',      # All caps headers
            r'^([A-Z][a-z\s]+):',         # Title case with colon
            r'^\s*\*\s*([A-Z][a-z\s]+)',  # Bullet point headers
        ]
        current_section = None
        current_content = []

        for line in text.split('\n'):
            line_stripped = line.strip()
            is_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line_stripped)
                if match:
                    if current_section and current_content:
                        sections.append({
                            'section': current_section,
                            'content': '\n'.join(current_content).strip(),
                            'type': 'section'
                        })
                    current_section = match.group(1).strip()
                    current_content = []
                    is_header = True
                    break
            if not is_header and line_stripped:
                current_content.append(line)

        if current_section and current_content:
            sections.append({
                'section': current_section,
                'content': '\n'.join(current_content).strip(),
                'type': 'section'
            })

        if not sections and text.strip():
            sections.append({'section': 'document', 'content': text.strip(), 'type': 'document'})
        return sections

    def chunk_by_sentences(self, text: str, max_tokens: int) -> List[str]: # max_tokens passed from semantic_chunk
        if not text.strip():
            return []
        try:
            sentences = self.sent_tokenizer.tokenize(text)
            sentences = [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}. Falling back to regex.")
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        chunks, current_chunk, current_length = [], [], 0
        for sentence in sentences:
            slen = len(sentence)
            if current_length + slen > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_length = [sentence], slen
            else:
                current_chunk.append(sentence)
                current_length += slen
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def chunk_by_paragraphs(self, text: str, max_tokens: int) -> List[str]: # max_tokens passed from semantic_chunk
        if not text.strip():
            return []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks, current_chunk, current_length = [], [], 0

        for paragraph in paragraphs:
            plen = len(paragraph)
            if plen > max_tokens:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk, current_length = [], 0
                chunks.extend(self.chunk_by_sentences(paragraph, max_tokens)) # Pass max_tokens here
            elif current_length + plen > max_tokens and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk, current_length = [paragraph], plen
            else:
                current_chunk.append(paragraph)
                current_length += plen
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        return chunks

    def semantic_chunk(self, text: str, strategy: str = "hybrid", max_tokens: int = MAX_CHUNK_TOKENS) -> List[Tuple[str, Dict[str, Any]]]:
        chunks_with_metadata: List[Tuple[str, Dict[str, Any]]] = []
        if strategy == "sections":
            sections = self.extract_sections(text)
            for section in sections:
                sec_chunks = self.chunk_by_paragraphs(section['content'], max_tokens) # Pass max_tokens
                for i, chunk in enumerate(sec_chunks):
                    chunks_with_metadata.append((chunk, {
                        'strategy': 'sections',
                        'section': section['section'],
                        'section_type': section['type'],
                        'chunk_index': i,
                        'total_chunks': len(sec_chunks)
                    }))
        elif strategy == "paragraphs":
            chunks = self.chunk_by_paragraphs(text, max_tokens) # Pass max_tokens
            for i, chunk in enumerate(chunks):
                chunks_with_metadata.append((chunk, {
                    'strategy': 'paragraphs', 'chunk_index': i, 'total_chunks': len(chunks)
                }))
        elif strategy == "sentences":
            chunks = self.chunk_by_sentences(text, max_tokens) # Pass max_tokens
            for i, chunk in enumerate(chunks):
                chunks_with_metadata.append((chunk, {
                    'strategy': 'sentences', 'chunk_index': i, 'total_chunks': len(chunks)
                }))
        elif strategy == "recursive":
            # recursive_splitter is initialized with MAX_CHUNK_TOKENS and CHUNK_OVERLAP from config
            chunks = self.recursive_splitter.split_text(text) 
            for i, chunk in enumerate(chunks):
                chunks_with_metadata.append((chunk, {
                    'strategy': 'recursive', 'chunk_index': i, 'total_chunks': len(chunks)
                }))
        elif strategy == "hybrid":
            sections = self.extract_sections(text)
            if len(sections) > 1:
                for section in sections:
                    sec_chunks = self.chunk_by_paragraphs(section['content'], max_tokens) # Pass max_tokens
                    for i, chunk in enumerate(sec_chunks):
                        chunks_with_metadata.append((chunk, {
                            'strategy': 'hybrid_sections',
                            'section': section['section'],
                            'section_type': section['type'],
                            'chunk_index': i,
                            'section_total_chunks': len(sec_chunks)
                        }))
            else:
                chunks = self.chunk_by_paragraphs(text, max_tokens) # Pass max_tokens
                for i, chunk in enumerate(chunks):
                    chunks_with_metadata.append((chunk, {
                        'strategy': 'hybrid_paragraphs', 'chunk_index': i, 'total_chunks': len(chunks)
                    }))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for i, (chunk, meta) in enumerate(chunks_with_metadata):
            meta.update({'global_chunk_id': i, 'chunk_length': len(chunk), 'max_tokens': max_tokens})
        return chunks_with_metadata

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        try:
            sentences = self.sent_tokenizer.tokenize(text)
            entities: Dict[str, List[str]] = {}
            for sentence in sentences:
                tokens = nltk.word_tokenize(sentence)
                pos_tags = nltk.pos_tag(tokens)
                ne_tree = nltk.ne_chunk(pos_tags, binary=False)
                for subtree in ne_tree:
                    if hasattr(subtree, 'label'):
                        entity_type = subtree.label()
                        entity_text = ' '.join(word for word, tag in subtree)
                        entities.setdefault(entity_type, [])
                        if entity_text not in entities[entity_type]:
                            entities[entity_type].append(entity_text)
            return entities
        except Exception as e:
            logger.warning(f"NLTK entity extraction failed: {e}")
            return {}

    def classify_content_type(self, text: str) -> str:
        tl = text.lower()
        if any(k in tl for k in ['indication','dosage','administration','contraindication','warning','adverse reaction','drug interaction','overdosage']):
            return 'drug_label'
        if any(k in tl for k in ['clinical trial','efficacy','safety','placebo','double-blind']):
            return 'clinical_data'
        if any(k in tl for k in ['fda','approval','regulation','compliance','inspection']):
            return 'regulatory'
        if any(k in tl for k in ['procedure','protocol','standard operating','sop','step']):
            return 'sop'
        return 'general'