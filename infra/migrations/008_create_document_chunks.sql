-- Migration: Create lexi_chunks table for pgvector similarity search (Cohere 1024-dim)
-- This table is shared by the AI Monolith, Ingestion Service, and Retrieval Service.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create ai schema
CREATE SCHEMA IF NOT EXISTS ai;

-- Document chunks table (Cohere embed-multilingual-v3.0 = 1024 dimensions)
CREATE TABLE IF NOT EXISTS ai.lexi_chunks (
    id VARCHAR PRIMARY KEY,
    doc_id VARCHAR NOT NULL,
    course VARCHAR NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    source VARCHAR NOT NULL DEFAULT 'uploaded_note',
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookup by doc and course
CREATE INDEX IF NOT EXISTS idx_lexi_chunks_doc_id ON ai.lexi_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_lexi_chunks_course ON ai.lexi_chunks(course);
