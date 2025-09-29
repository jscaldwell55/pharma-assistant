# backend/core_logic/semantic_chunker.py
import nltk
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import logging
import tiktoken

from .config import settings
MAX_CHUNK_TOKENS = settings.chunking.MAX_CHUNK_TOKENS
CHUNK_OVERLAP = settings.chunking.CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Ensure required NLTK data
def _ensure_nltk_resource(find_path: str, download_name: str):
    try:
        nltk.data.find(find_path)
    except LookupError:
        nltk.download(download_name)

_ensure_nltk_resource('tokenizers/punkt', 'punkt')
_ensure_nltk_resource('chunkers/maxent_ne_chunker', 'maxent_ne_chunker')
_ensure_nltk_resource('corpora/words', 'words')
_ensure_nltk_resource('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')

# --- Tokenizer for token-aware chunk sizing ---
_enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))

class SemanticChunker:
    """
    Semantic chunking for medical documents using NLTK + langchain.
    Preserves structure and uses token-based length limits.
    """
    def __init__(self):
        self.sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_TOKENS,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        sections = []
        section_patterns = [
            r'^\s*\d+\.?\s+([A-Z\s]+)$',         # Numbered sections
            r'^\d+(\.\d+)+\s+[A-Z][A-Z\s]+$',    # Numbered subsections (e.g., 5.1 WARNINGS)
            r'^([A-Z][A-Z\s]{3,})$',             # All caps headers
            r'^([A-Z][a-z\s]+):',                # Title case with colon
            r'^\s*\*\s*([A-Z][a-z\s]+)',         # Bullet point headers
        ]
        current_section, current_content = None, []

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

    def chunk_by_sentences(self, text: str, max_tokens: int) -> List[str]:
        if not text.strip():
            return []
        try:
            sentences = self.sent_tokenizer.tokenize(text)
            sentences = [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"Sentence tokenizer failed: {e}. Falling back to regex.")
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        chunks, current_chunk, current_len = [], [], 0
        for sentence in sentences:
            slen = count_tokens(sentence)
            if current_len + slen > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_len = [sentence], slen
            else:
                current_chunk.append(sentence)
                current_len += slen
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def chunk_by_paragraphs(self, text: str, max_tokens: int) -> List[str]:
        if not text.strip():
            return []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks, current_chunk, current_len = [], [], 0

        for paragraph in paragraphs:
            plen = count_tokens(paragraph)
            if plen > max_tokens:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk, current_len = [], 0
                chunks.extend(self.chunk_by_sentences(paragraph, max_tokens))
            elif current_len + plen > max_tokens and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk, current_len = [paragraph], plen
            else:
                current_chunk.append(paragraph)
                current_len += plen
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        return chunks

    def semantic_chunk(self, text: str, strategy: str = "hybrid", max_tokens: int = MAX_CHUNK_TOKENS) -> List[Tuple[str, Dict[str, Any]]]:
        chunks_with_metadata: List[Tuple[str, Dict[str, Any]]] = []

        if strategy == "sections":
            sections = self.extract_sections(text)
            for section in sections:
                sec_chunks = self.chunk_by_paragraphs(section['content'], max_tokens)
                for i, chunk in enumerate(sec_chunks):
                    chunks_with_metadata.append((chunk, {
                        'strategy': 'sections',
                        'section': section['section'],
                        'section_type': section['type'],
                        'chunk_index': i,
                        'total_chunks': len(sec_chunks)
                    }))

        elif strategy == "paragraphs":
            chunks = self.chunk_by_paragraphs(text, max_tokens)
            for i, chunk in enumerate(chunks):
                chunks_with_metadata.append((chunk, {
                    'strategy': 'paragraphs',
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }))

        elif strategy == "sentences":
            chunks = self.chunk_by_sentences(text, max_tokens)
            for i, chunk in enumerate(chunks):
                chunks_with_metadata.append((chunk, {
                    'strategy': 'sentences',
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }))

        elif strategy == "recursive":
            chunks = self.recursive_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunks_with_metadata.append((chunk, {
                    'strategy': 'recursive',
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }))

        elif strategy == "hybrid":
            sections = self.extract_sections(text)
            if len(sections) > 1:
                for section in sections:
                    sec_chunks = self.chunk_by_paragraphs(section['content'], max_tokens)
                    for i, chunk in enumerate(sec_chunks):
                        chunks_with_metadata.append((chunk, {
                            'strategy': 'hybrid_sections',
                            'section': section['section'],
                            'section_type': section['type'],
                            'chunk_index': i,
                            'section_total_chunks': len(sec_chunks)
                        }))
            else:
                chunks = self.chunk_by_paragraphs(text, max_tokens)
                for i, chunk in enumerate(chunks):
                    chunks_with_metadata.append((chunk, {
                        'strategy': 'hybrid_paragraphs',
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for i, (chunk, meta) in enumerate(chunks_with_metadata):
            meta.update({'global_chunk_id': i, 'token_length': count_tokens(chunk), 'max_tokens': max_tokens})
        return chunks_with_metadata
