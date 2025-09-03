# backend/core_logic/generate_chunks_from_pdfs.py
# Usage:
#   pip install pypdf
#   python generate_chunks_from_pdfs.py --pdf_dir ./data/pdfs --out ./data/chunks/chunks.jsonl --doc Journavx

import argparse, os, json, re
from pathlib import Path
from typing import List
from pypdf import PdfReader

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_paragraphs(paragraphs: List[str], max_chars: int = 1200) -> List[str]:
    chunks, buf = [], ""
    for p in paragraphs:
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf}\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf.strip())
            buf = p
    if buf:
        chunks.append(buf.strip())
    return chunks

def pdf_to_paragraphs(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    paras: List[str] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # split on blank lines, fallback to long lines
        raw_paras = re.split(r"\n\s*\n", text.strip()) if text else []
        page_paras = [normalize_ws(p) for p in raw_paras if normalize_ws(p)]
        if not page_paras and text:
            page_paras = [normalize_ws(text)]
        paras.extend(page_paras)
    return paras

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Folder containing PDFs")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--doc", default="Document", help="Logical document name (for IDs/metadata)")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        paragraphs = pdf_to_paragraphs(pdf)
        chunks = chunk_paragraphs(paragraphs, max_chars=1200)
        for i, chunk in enumerate(chunks, start=1):
            rec = {
                "id": f"{args.doc}:{pdf.stem}:{i:04d}",
                "text": chunk,
                "meta": {
                    "source": pdf.name,
                    "doc": args.doc,
                }
            }
            records.append(rec)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} chunks → {out_path}")

if __name__ == "__main__":
    main()
