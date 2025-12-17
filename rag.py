import PyPDF2
import os
import logging
from google import genai
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import sentence_transformers



load_dotenv()
logger = logging.getLogger()
Hf_token=os.environ["HF_TOKEN"]

def load_local_kb(path="/Users/thumatisireesha/Desktop/CHATBOT/data/project.pdf"):
    pages = []
    try:
        with open(path, "rb") as f:  # open in binary mode
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                text=p.extract_text()
                if text and text.strip():
                    pages.append(text.strip())
            logger.info("Loaded %d pages from local KB", len(pages))
    except Exception as e:
        logger.warning("Could not load local KB: %s", e)
    return pages
    
# chunk text 
def chunk_text(chunk_size=300):
    pages=load_local_kb()
    chunks = []
    for page in pages:
        words = page.split()
        if not words:
            continue
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks

# Initialize Pinecone

# pinecone_api_key = os.getenv("PINECONE_API_KEY")

# index_name = "chatbot-kb"
# embedding_dimension = 384

# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name, 
#         dimension=embedding_dimension,
#         metric="cosine",
#         spec = ServerlessSpec(
#             cloud="aws",
#             region="us-east-1" )
# )

# index = pc.Index(index_name)

def embedding_vector(texts):
    if not texts:
        logger.warning("No texts provided for embedding")
        return [], []
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    embedding_list = embeddings.embed_documents(texts)
    print(len(embedding_list))
    return texts,embedding_list
   
def init_pinecone_index(texts,vectors):
    pc = Pinecone(api_key="PINECONE_API_KEY")
    if not pc:
        raise RuntimeError("PINECONE_API_KEY not set in environment")
    existing = pc.list_indexes()
    if isinstance(existing, list):
        index_names = existing
    else:
        # fallback try .names()
        try:
            index_names = existing.names()
        except Exception:
            index_names = existing
    if "rag-kb"not in index_names:
        logger.info("Creating index %s with dimension=%d", "rag-kb", 384)
        pc.create_index(name="rag-kb", dimension=384, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    index = pc.Index("rag-kb")
    # Validate embedding dims
    dims = {len(v) for v in vectors}
    if len(dims) != 1 or next(iter(dims)) != 384:
        raise ValueError(f"Embedding dimension mismatch: found dims={dims}, expected {384}")

    # Convert to plain float lists
    clean_embeddings = [list(map(float, v)) for v in vectors]
    # ✅ YOUR UPSERT LINES GO HERE
    # -------------------------------
    upsert_items = [
        (str(i), clean_embeddings[i], {"text": texts[i]})
        for i in range(len(clean_embeddings))
    ]

    print("Upserting", len(upsert_items), "vectors into Pinecone...")
    index.upsert(upsert_items)
    print("Upsert DONE.")





if __name__ == "__main__":
    texts = chunk_text(chunk_size=500)
    if not texts:
        print("No chunks to embed. Exiting.")
        raise SystemExit(1)
    texts, vectors = embedding_vector(texts)
    print("Embedding completed.")
    print("Number of chunks:", len(texts))
    print("Number of vectors returned:", len(vectors))
    if vectors:
        print("Vector dimension (first vector):", len(vectors[0]))
    init_pinecone_index(texts,vectors)

