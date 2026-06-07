import pathlib
import sys
from dotenv import load_dotenv

from src.core.db import store_chunks, upsert_document
from src.ingestion.docling_parser import parse_document

load_dotenv()

_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 300


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split long text into overlapping chunks.

    Tables and image captions should not be split.
    """
    chunks: list[str] = []

    if not text:
        return chunks

    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def _normalize_chunk(elem: dict, document_name: str, product_category: str) -> dict:
    """
    Normalize parser output into the format expected by Smart Banking Assistant.
    """

    metadata = elem.get("metadata", {}) or {}

    chunk_type = (
        elem.get("chunk_type")
        or elem.get("content_type")
        or "text"
    )

    return {
        "content": elem.get("content", ""),
        "chunk_type": chunk_type,
        "metadata": {
            **metadata,
            "document_name": document_name,
            "product_category": product_category,
            "source_page": metadata.get("source_page") or metadata.get("page_number"),
        },
    }


def run_ingestion(file_path: str, product_category: str = "general") -> dict:
    """
    Run ingestion for one banking document.

    Steps:
    1. Register document
    2. Parse using docling_parser
    3. Split only long text chunks
    4. Store chunks into PGVector table
    """

    resolved = pathlib.Path(file_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    document_name = resolved.name

    doc_id = upsert_document(document_name, str(resolved))

    print(f"[ingestion] doc_id={doc_id}")
    print(f"[ingestion] file={resolved}")
    print(f"[ingestion] product_category={product_category}")

    parsed_elements = parse_document(str(resolved))

    print(f"[ingestion] Parsed elements: {len(parsed_elements)}")

    chunks: list[dict] = []

    for elem in parsed_elements:
        normalized = _normalize_chunk(
            elem=elem,
            document_name=document_name,
            product_category=product_category,
        )

        content = normalized["content"]
        chunk_type = normalized["chunk_type"]

        if chunk_type == "text" and len(content) > _TEXT_CHUNK_SIZE:
            sub_chunks = _split_text(
                content,
                _TEXT_CHUNK_SIZE,
                _TEXT_CHUNK_OVERLAP,
            )

            for sub in sub_chunks:
                chunks.append({
                    "content": sub,
                    "chunk_type": "text",
                    "metadata": normalized["metadata"],
                })
        else:
            chunks.append(normalized)

    print(f"[ingestion] Chunks ready for embedding: {len(chunks)}")

    count = store_chunks(chunks, doc_id)

    print(f"[ingestion] Stored chunks: {count}")

    return {
        "status": "success",
        "doc_id": str(doc_id),
        "chunks_ingested": count,
        "document_name": document_name,
        "product_category": product_category,
    }


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        pdf_path = pathlib.Path(sys.argv[1])
    else:
        pdf_path = pathlib.Path("data/sample_docs/home_loan_policy.pdf")

    if len(sys.argv) >= 3:
        category = sys.argv[2]
    else:
        category = "loan"

    result = run_ingestion(str(pdf_path), product_category=category)

    print("\nIngestion complete:")
    print(result)