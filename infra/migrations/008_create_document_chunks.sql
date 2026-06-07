-- Migration: Create document chunks table for pgvector similarity search
-- Enables the vector extension and defines the schema for RAG document chunk storage.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Document chunks table
CREATE TABLE IF NOT EXISTS public.document_chunks (
    id VARCHAR PRIMARY KEY,
    material_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    chunk_index INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookup by user and material
CREATE INDEX IF NOT EXISTS idx_document_chunks_material_id ON public.document_chunks(material_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_user_id ON public.document_chunks(user_id);
