"""
Data ingestion pipeline for the RAG system.

Reads scraped bank data from Data/raw/<bank>/{loans,deposits,branches}.json,
chunks it into embeddable segments, generates Armenian text embeddings,
and stores everything in a ChromaDB persistent vector database.

The agent (agent.py) later queries this database to retrieve relevant
context for answering user questions.

Chunking strategy:
    - SHORT items (description < 1000 chars): grouped together into chunks
      up to 1500 chars each. This includes branches, simple loan/deposit
      product summaries (mostly Evoca's compact format).
    - LONG items (description >= 1000 chars): chunked individually at
      sentence boundaries. This includes ACBA/Ameria product pages with
      full tab content, and Evoca's legal info pages. Each chunk gets a
      header with the product's key fields (bank, name, etc.) for context.

Embedding model: Metric-AI/armenian-text-embeddings-2-base
    - Trained specifically for Armenian text
    - Produces 768-dim vectors
    - Used for both ingestion and query-time retrieval

ChromaDB collection: "bank_data"
    - Metadata: {"bank": "evoca"|"ameria"|"acba", "topic": "loans"|"deposits"|"branches"}
    - Used by agent.py to filter retrieval by topic

Usage:
    python ingest.py

    Must be run after scraping (scraper.py) and before running the agent.
    Re-running deletes and recreates the entire collection.

Data flow:
    Data/raw/evoca/loans.json  ─┐
    Data/raw/ameria/loans.json ─┤──> ingest.py ──> Data/chroma_db/
    Data/raw/acba/loans.json   ─┘
    (same for deposits.json and branches.json)
"""

import json
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb

# load Metric Armenian embeddings
MODEL_NAME = "Metric-AI/armenian-text-embeddings-2-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# setup ChromaDB, recreate to clear stale giant chunks
client = chromadb.PersistentClient(path="Data/chroma_db")
try:
    client.delete_collection("bank_data")
    print("Cleared old collection.")
except Exception:
    pass
collection = client.get_or_create_collection("bank_data")

MAX_CHUNK_CHARS = 1500
# Items with description longer than this get chunked individually
LONG_THRESHOLD = 1000


def embed(text):
    """
    Generate a 768-dimensional embedding vector for Armenian text.
    Uses CLS token pooling (first token of last hidden state).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].squeeze().tolist()


def normalize_armenian_text(text):
    """Fix mashed-together sentences from BeautifulSoup's get_text(strip=True)."""
    # Insert newline after colon/Armenian stop + uppercase Armenian letter
    text = re.sub(r'([:։])([\u0531-\u0556])', r'\1\n\2', text)
    # Insert space between lowercase Armenian + uppercase Armenian
    text = re.sub(r'([\u0561-\u0586])([\u0531-\u0556])', r'\1 \2', text)
    return text


def split_text(text, max_chars=MAX_CHUNK_CHARS):
    """Split long text into smaller chunks at sentence boundaries."""
    text = normalize_armenian_text(text)
    if len(text) <= max_chars:
        return [text]

    # Split at sentence boundaries
    sentences = re.split(r'(?<=։)\s*|(?<=:)\s*\n|(?<=\.)\s+|\n+', text)

    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Handle sentences that are themselves too long
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            
            # Hard-split at max_chars boundaries
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i : i + max_chars])
            continue
        
        # Try to add sentence to current chunk
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence) if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:max_chars]]


def compact(item):
    """Convert a JSON item to a compact text representation (ALL fields except url/type)."""
    parts = []
    for k, v in item.items():
        if k in ("url", "type") or not v:
            continue
        parts.append(f"{k}: {v}")
    return " | ".join(parts)


def compact_short(item):
    """Compact representation WITHOUT the full description - just key fields."""
    parts = []
    for k, v in item.items():
        if k in ("url", "type", "description") or not v:
            continue
        parts.append(f"{k}: {v}")
    return " | ".join(parts)


def is_long_item(item):
    """
    Check if an item needs individual chunking (vs grouping with others).

    Items with description > 1000 chars are "long":
    - ACBA/Ameria product pages with full tab content (7-12K chars)
    - Evoca info pages (important-information, credit-history: 20-23K chars)

    Short items (branches, simple products) get grouped together.
    """
    desc = item.get("description", "")
    return len(desc) > LONG_THRESHOLD


def ingest():
    """
    Main ingestion function. Processes all banks and topics.

    For each JSON file in Data/raw/<bank>/:
    1. Separates items into short (< 1000 chars desc) and long (>= 1000 chars)
    2. Short items: compacted to one line each, grouped into chunks up to 1500 chars
       - Branches (17-66 items) -> multiple chunks of grouped entries
       - Simple products (3-4 items) -> often fits in 1 chunk
    3. Long items: description split at sentence boundaries into 1500-char chunks
       - First chunk gets full header (bank, name, key fields)
       - Subsequent chunks get minimal header (bank + name only)
    4. Each chunk is embedded and upserted into ChromaDB with metadata
    """
    banks_dir = Path("Data/raw")
    total = 0

    for bank_dir in sorted(banks_dir.iterdir()):
        if not bank_dir.is_dir():
            continue
        bank = bank_dir.name

        for topic_file in sorted(bank_dir.glob("*.json")):
            topic = topic_file.stem
            print(f"Ingesting {bank}/{topic}...")
            data = json.load(open(topic_file, encoding="utf-8"))

            if not isinstance(data, list):
                continue

            # Split into short items (group together) vs long items (chunk individually)
            short_items = [item for item in data if not is_long_item(item)]
            long_items = [item for item in data if is_long_item(item)]

            # SHORT ITEMS: group together into chunks (branches, simple products)
            if short_items:
                product_texts = [compact(p) for p in short_items]
                combined = "\n".join(product_texts)

                if len(combined) <= MAX_CHUNK_CHARS:
                    # Everything fits in one chunk
                    doc_id = f"{bank}_{topic}_products"
                    collection.upsert(
                        ids=[doc_id],
                        embeddings=[embed(combined)],
                        documents=[combined],
                        metadatas=[{"bank": bank, "topic": topic}],
                    )
                    print(f"  {len(short_items)} short items -> 1 chunk ({len(combined)} chars)")
                    total += 1
                else:
                    # Split into multiple chunks, grouping items greedily
                    chunk_texts = []
                    current = ""
                    for pt in product_texts:
                        if len(current) + len(pt) + 1 > MAX_CHUNK_CHARS and current:
                            chunk_texts.append(current)
                            current = pt
                        else:
                            current = (current + "\n" + pt) if current else pt
                    if current:
                        chunk_texts.append(current)

                    for i, chunk in enumerate(chunk_texts):
                        doc_id = f"{bank}_{topic}_short_{i}"
                        collection.upsert(
                            ids=[doc_id],
                            embeddings=[embed(chunk)],
                            documents=[chunk],
                            metadatas=[{"bank": bank, "topic": topic}],
                        )
                    print(f"  {len(short_items)} short items -> {len(chunk_texts)} chunks")
                    total += len(chunk_texts)

            # LONG ITEMS: chunk each one individually 
            for item in long_items:
                name = item.get("name", "unknown")
                description = item.get("description", "")
                if not description:
                    continue

                # Build header from key fields (everything except description)
                header = compact_short(item)
                if not header:
                    header = f"Bank: {bank} | Topic: {topic} | Name: {name}"

                # Split description into sentence-boundary chunks
                parts = split_text(description, MAX_CHUNK_CHARS)
                for i, part in enumerate(parts):
                    doc_id = f"{bank}_{topic}_{name[:50]}_{i}"
                    # First chunk gets full header, rest get minimal context
                    if i == 0:
                        text = f"{header}\n{part}"
                    else:
                        text = f"Bank: {bank} | {name}\n{part}"
                    collection.upsert(
                        ids=[doc_id],
                        embeddings=[embed(text)],
                        documents=[text],
                        metadatas=[{"bank": bank, "topic": topic}],
                    )
                print(f"  {name[:60]}... -> {len(parts)} chunks")
                total += len(parts)

    print(f"\nTotal: {total} chunks ingested.")


if __name__ == "__main__":
    ingest()
    print("Done.")