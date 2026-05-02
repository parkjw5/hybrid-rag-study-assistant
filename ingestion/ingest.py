import numpy as np
from pathlib import Path
import re
import faiss

from config import PDF_FOLDER, FAISS_INDEX_PATH
from storage.database import Database
from core.chunker import chunk_text
from core.embedding import EmbeddingEngine
from ingestion.pdf_utils import extract_text_from_pdf


# -----------------------------------
# Clean text helper
# -----------------------------------
def clean_text(text):
    if not text:
        return ""

    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------------
# Main Ingestion Function
# -----------------------------------
def ingest_all_pdfs():

    pdf_folder = Path(PDF_FOLDER)
    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDFs found in data/pdfs/")

    print(f"Found {len(pdf_files)} PDF(s). Starting ingestion...")

    db = Database()
    engine = EmbeddingEngine()

    all_embeddings = []

    # -----------------------------------
    # Process each PDF
    # -----------------------------------
    for pdf_path in pdf_files:

        print(f"Ingesting: {pdf_path.name}")

        document_id = db.insert_document(pdf_path.stem)
        pages = extract_text_from_pdf(str(pdf_path))

        for page_number, page_text in pages:

            page_text = clean_text(page_text)
            chunks = chunk_text(page_text)

            for chunk in chunks:
                if not chunk.strip():
                    continue

                db.insert_chunk(
                    document_id=document_id,
                    content=chunk,
                    page_number=page_number,
                )

                embedding = engine.embed_documents([chunk])[0]
                all_embeddings.append(embedding)

    if not all_embeddings:
        raise ValueError("No embeddings generated. Check PDFs.")

    print(f"Generated {len(all_embeddings)} embeddings.")

    # -----------------------------------
    # Build FAISS Index
    # -----------------------------------
    embeddings_array = np.array(all_embeddings).astype("float32")

    dimension = engine.dimension
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)

    Path("data").mkdir(exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("FAISS index saved.")
    print("Ingestion complete.")

    db.close()


if __name__ == "__main__":
    ingest_all_pdfs()