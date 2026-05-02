import re
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP):
    """
    Sentence-aware chunking.
    Avoids cutting sentences mid-way.
    Chunk size measured in characters.
    """

    if not text:
        return []

    # Split into sentences (simple heuristic)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())

            # Apply overlap from previous chunk
            overlap_text = current_chunk[-overlap:]
            current_chunk = overlap_text + sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks