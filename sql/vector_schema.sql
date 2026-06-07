CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT UNIQUE NOT NULL,
    source_path TEXT,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS multimodal_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,

    chunk_type VARCHAR(50),
    element_type VARCHAR(100),
    content TEXT,

    image_path TEXT,
    mime_type VARCHAR(100),

    page_number INT,
    section TEXT,
    source_file TEXT,
    document_name TEXT,
    product_category VARCHAR(50),

    position JSONB,
    embedding VECTOR(1536),
    metadata JSONB,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_multimodal_chunks_doc_id
ON multimodal_chunks(doc_id);

CREATE INDEX IF NOT EXISTS idx_multimodal_chunks_chunk_type
ON multimodal_chunks(chunk_type);

CREATE INDEX IF NOT EXISTS idx_multimodal_chunks_product_category
ON multimodal_chunks(product_category);

CREATE INDEX IF NOT EXISTS idx_multimodal_chunks_embedding
ON multimodal_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);