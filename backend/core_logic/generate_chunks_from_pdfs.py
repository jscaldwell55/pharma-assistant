#!/usr/bin/env python3
"""
Generate chunks from PDF documents for Pinecone indexing.
This script processes PDFs and creates JSONL files with chunks ready for embedding.
"""

import os
import sys
import json
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

# Import your chunking logic
from .semantic_chunker import SemanticChunker

# PDF processing
try:
    import pymupdf  # fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # PyMuPDF alternative import
        import pymupdf
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF files and extract text"""
    
    def __init__(self):
        if not (PYMUPDF_AVAILABLE or PYPDF_AVAILABLE):
            raise ImportError(
                "No PDF library available. Install either:\n"
                "  pip install pymupdf\n"
                "  OR\n"
                "  pip install pypdf"
            )
        self.use_pymupdf = PYMUPDF_AVAILABLE
        logger.info(f"Using PDF library: {'PyMuPDF' if self.use_pymupdf else 'pypdf'}")
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract all text from a PDF file"""
        if self.use_pymupdf:
            return self._extract_with_pymupdf(pdf_path)
        else:
            return self._extract_with_pypdf(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (better for complex layouts)"""
        import fitz
        text_parts = []
        
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf, 1):
                text = page.get_text()
                if text.strip():
                    # Add page marker for reference
                    text_parts.append(f"[Page {page_num}]\n{text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf(self, pdf_path: str) -> str:
        """Extract text using pypdf"""
        text_parts = []
        
        with open(pdf_path, 'rb') as file:
            pdf = PdfReader(file)
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text.strip():
                    # Add page marker for reference
                    text_parts.append(f"[Page {page_num}]\n{text}")
        
        return "\n\n".join(text_parts)

def generate_chunk_id(text: str, doc_id: str, chunk_index: int) -> str:
    """Generate a unique ID for a chunk"""
    content = f"{doc_id}:{chunk_index}:{text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()

def process_pdf_to_chunks(
    pdf_path: str,
    output_path: str,
    chunking_strategy: str = "hybrid",
    max_chunk_size: int = 1500
) -> int:
    """
    Process a single PDF file into chunks and save to JSONL
    
    Args:
        pdf_path: Path to PDF file
        output_path: Path to output JSONL file
        chunking_strategy: One of 'hybrid', 'sections', 'paragraphs', 'sentences', 'recursive'
        max_chunk_size: Maximum characters per chunk
    
    Returns:
        Number of chunks created
    """
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Extract text
    processor = PDFProcessor()
    text = processor.extract_text(pdf_path)
    
    if not text.strip():
        logger.warning(f"No text extracted from {pdf_path}")
        return 0
    
    logger.info(f"Extracted {len(text)} characters from PDF")
    
    # Initialize chunker
    chunker = SemanticChunker()
    
    # Generate chunks
    chunks_with_metadata = chunker.semantic_chunk(
        text,
        strategy=chunking_strategy,
        max_tokens=max_chunk_size
    )
    
    logger.info(f"Created {len(chunks_with_metadata)} chunks")
    
    # Prepare document metadata
    doc_name = Path(pdf_path).stem
    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:12]
    
    # Format chunks for output
    output_chunks = []
    for idx, (chunk_text, chunk_meta) in enumerate(chunks_with_metadata):
        # Skip empty chunks
        if not chunk_text.strip():
            continue
        
        chunk_id = generate_chunk_id(chunk_text, doc_id, idx)
        
        output_chunk = {
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "source": doc_name,
                "doc_id": doc_id,
                "chunk_id": idx,
                "total_chunks": len(chunks_with_metadata),
                "chunk_strategy": chunk_meta.get("strategy", chunking_strategy),
                "section": chunk_meta.get("section", ""),
                "chunk_length": len(chunk_text),
                **chunk_meta
            }
        }
        output_chunks.append(output_chunk)
    
    # Write to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in output_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(output_chunks)} chunks to {output_path}")
    return len(output_chunks)

def process_directory(
    input_dir: str = "backend/core_logic/data",
    output_file: str = "backend/core_logic/data/chunks.jsonl",
    chunking_strategy: str = "hybrid",
    max_chunk_size: int = 1500
) -> None:
    """
    Process all PDFs in a directory
    
    Args:
        input_dir: Directory containing PDF files
        output_file: Output JSONL file path
        chunking_strategy: Chunking strategy to use
        max_chunk_size: Maximum chunk size in characters
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    all_chunks = []
    for pdf_path in pdf_files:
        try:
            # Process to temporary file first
            temp_output = str(pdf_path).replace('.pdf', '_chunks.jsonl')
            
            num_chunks = process_pdf_to_chunks(
                str(pdf_path),
                temp_output,
                chunking_strategy,
                max_chunk_size
            )
            
            # Read chunks and add to collection
            if num_chunks > 0:
                with open(temp_output, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_chunks.append(json.loads(line))
                
                # Clean up temp file
                os.remove(temp_output)
            
            logger.info(f"Successfully processed {pdf_path.name}: {num_chunks} chunks")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            continue
    
    # Write all chunks to final output file
    if all_chunks:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Successfully created {output_path} with {len(all_chunks)} total chunks")
    else:
        logger.warning("No chunks were generated from any PDFs")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate chunks from PDF documents")
    parser.add_argument(
        "--input-dir",
        default="backend/core_logic/data",
        help="Directory containing PDF files (default: backend/core_logic/data)"
    )
    parser.add_argument(
        "--output",
        default="backend/core_logic/data/chunks.jsonl",
        help="Output JSONL file (default: backend/core_logic/data/chunks.jsonl)"
    )
    parser.add_argument(
        "--strategy",
        choices=["hybrid", "sections", "paragraphs", "sentences", "recursive"],
        default="hybrid",
        help="Chunking strategy (default: hybrid)"
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=1500,
        help="Maximum chunk size in characters (default: 1500)"
    )
    parser.add_argument(
        "--single-pdf",
        help="Process a single PDF file instead of directory"
    )
    
    args = parser.parse_args()
    
    if args.single_pdf:
        # Process single PDF
        output = args.output
        if output.endswith('.jsonl'):
            # If output is a JSONL file, use it directly
            pass
        else:
            # Otherwise, generate output filename
            output = args.single_pdf.replace('.pdf', '_chunks.jsonl')
        
        num_chunks = process_pdf_to_chunks(
            args.single_pdf,
            output,
            args.strategy,
            args.max_chunk_size
        )
        print(f"✅ Generated {num_chunks} chunks from {args.single_pdf}")
    else:
        # Process directory
        process_directory(
            args.input_dir,
            args.output,
            args.strategy,
            args.max_chunk_size
        )

if __name__ == "__main__":
    main()