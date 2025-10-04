# Project 13: Vector Databases with Chroma

## Overview
Learn to work with vector databases for semantic search, document retrieval, and building RAG (Retrieval-Augmented Generation) applications using ChromaDB.

## Project Structure
```
13 - Vector Database/
└── chroma.ipynb    # ChromaDB tutorial and examples
```

## Learning Objectives
- Vector database fundamentals
- Embedding generation
- Semantic search implementation
- ChromaDB operations
- RAG architecture
- Similarity search techniques

## Getting Started

### Prerequisites
```bash
pip install chromadb openai sentence-transformers
pip install langchain  # Optional, for advanced RAG
```

### How to Run
```bash
jupyter notebook chroma.ipynb
```

## Key Concepts

### Vector Databases
- Store high-dimensional embeddings
- Enable semantic search
- Fast similarity lookups
- Scalable document retrieval

### ChromaDB Features
- **Simple API**: Easy to use
- **Persistent Storage**: Save collections
- **Metadata Filtering**: Advanced queries
- **Multiple Embeddings**: Support various models
- **Local First**: No external dependencies

## Use Cases

### 1. Semantic Search
- Find similar documents
- Question answering
- Content recommendation
- Duplicate detection

### 2. RAG Applications
- Chat with documents
- Knowledge bases
- Q&A systems
- Context-aware AI

### 3. Similarity Detection
- Plagiarism detection
- Content clustering
- Recommendation engines
- Image/text matching

## Core Operations
```python
# Create collection
collection = client.create_collection("documents")

# Add documents
collection.add(
    documents=["text1", "text2"],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=5
)
```

## Embedding Models
- OpenAI embeddings
- Sentence Transformers
- Custom embeddings
- Multi-language models

## Applications
- **Document Search**: Semantic document retrieval
- **Chatbots**: Context-aware responses
- **Recommendation**: Similar content suggestion
- **Q&A Systems**: Knowledge base queries
- **Research**: Academic paper search
- **E-commerce**: Product recommendations

## Advanced Features
- Metadata filtering
- Hybrid search (keyword + semantic)
- Multi-modal embeddings
- Collection management
- Batch operations
- Distance metrics (cosine, L2, IP)

## Integration with LLMs
- Provide relevant context
- Reduce hallucinations
- Ground responses in data
- Dynamic knowledge updates
- Cost-effective scaling
