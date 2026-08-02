#  LangChain RAG (Retrieval-Augmented Generation)

# Session Summary: A complete RAG implementation using LangChain, HuggingFace embeddings, and FAISS vector store for grounded question answering.


## What I Did During the Session

1. **Implemented Document Loading**:
   - Used `TextLoader`, `DirectoryLoader`, and `PyPDFLoader` to load documents from various sources
   - Created a flexible `load_documents()` function supporting txt, pdf, and directory inputs
   - Documents are loaded as LangChain `Document` objects with content and metadata

2. **Set Up Document Chunking**:
   - Used `RecursiveCharacterTextSplitter` to split documents into smaller chunks
   - Configured `chunk_size=1000` and `chunk_overlap=200` for context preservation
   - Overlap ensures important context isn't lost at chunk boundaries

3. **Created Embeddings with HuggingFace**:
   - Used `HuggingFaceEmbeddings` with the `sentence-transformers/all-MiniLM-L6-v2` model
   - Configured embedding normalization for better similarity search results
   - Embeddings convert text into numerical vectors for semantic comparison

4. **Built a FAISS Vector Store**:
   - Used `FAISS.from_documents()` to create a vector store from document chunks
   - Implemented persistence with `save_local()` and `load_local()` for reuse
   - FAISS enables efficient similarity search over embedded documents

5. **Created a RAG Chain with LCEL**:
   - Built a retriever from the vector store with `as_retriever(search_type="similarity")`
   - Designed a grounding prompt template instructing the LLM to only use provided context
   - Chained components: `retriever → format_docs → prompt → llm → StrOutputParser()`
   - Used `RunnablePassthrough()` to pass the question through the chain

6. **Implemented Question Answering with Source Attribution**:
   - Created `ask_question()` function that retrieves relevant documents and generates answers
   - Added source document display to verify answers are grounded in documents

---

## What I Learned

1. **RAG Architecture**:
   - **RAG (Retrieval-Augmented Generation)** combines retrieval systems with generative models
   - The pattern: Query → Retrieve relevant docs → Augment prompt with context → Generate answer
   - RAG prevents hallucination by grounding LLM responses in actual document content

2. **Document Processing Pipeline**:
   - **Document Loaders** standardize loading from various sources (files, directories, PDFs)
   - **Text Splitters** break documents into chunks that fit within LLM context windows
   - `chunk_overlap` preserves context across chunk boundaries for better retrieval

3. **Vector Embeddings and Semantic Search**:
   - **Embeddings** convert text into dense vectors capturing semantic meaning
   - Similar concepts have similar vector representations (close in vector space)
   - `sentence-transformers/all-MiniLM-L6-v2` is a lightweight, efficient embedding model
   - **Similarity search** finds documents closest to the query in vector space

4. **FAISS Vector Store**:
   - **FAISS (Facebook AI Similarity Search)** is an efficient library for similarity search
   - Supports persistence to disk for reuse without re-embedding documents
   - `as_retriever()` converts a vector store into a LangChain retriever component

5. **Building Grounded RAG Chains**:
   - **Grounding prompts** explicitly instruct the LLM to only use provided context
   - `RunnablePassthrough()` passes input unchanged through the chain
   - `StrOutputParser()` extracts the string output from the LLM response
   - Lower `temperature` (e.g., 0.3) produces more factual, deterministic responses

6. **RAG Best Practices**:
   - Always show source documents to verify answer grounding
   - Instruct the LLM to say "I don't know" when context is insufficient
   - Use `k` parameter to control how many documents are retrieved
   - Normalize embeddings for better similarity comparisons

---

This session provided hands-on experience building a complete RAG pipeline from scratch — loading and chunking documents, embedding them with HuggingFace models, indexing in a FAISS vector store, and composing an LCEL chain that grounds LLM answers in retrieved context with source attribution.

---

# Session 2 — Reranking Retriever (Cross-Encoder)

# Session Summary: Upgraded the RAG retriever from a single bi-encoder pass to a two-stage retrieve-then-rerank pipeline, using a cross-encoder to reorder FAISS candidates for sharper relevance (`reranking_rag.py`).

## What I Did During the Session

1. **Identified the Bi-Encoder Limitation**:
   - The base RAG retrieves with a bi-encoder (query and docs embedded independently), which is fast but a coarse relevance signal
   - Built a small in-memory corpus with lexical "trap" documents that share vocabulary with the query ("memory", "deployment", "large model") without answering it

2. **Implemented Stage 1 — Wide Retrieval**:
   - Reused `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) + FAISS from Session 1
   - Used `similarity_search_with_score` to retrieve a wider candidate set (top-6) and printed the ranking with distances

3. **Implemented Stage 2 — Cross-Encoder Reranking**:
   - Loaded a `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) from `sentence-transformers`
   - Scored each `[query, document]` pair jointly and re-sorted, keeping the top-3
   - Printed a before → after comparison (`▲/▼ was #k`) so the reordering is visible

4. **Assembled Reranked Context**:
   - Joined the reranked top-3 into grounded context ready for the LLM generation step
   - Kept the demo retrieval-only so it runs CPU-only with no model download

---

## What I Learned

1. **Bi-Encoder vs. Cross-Encoder**:
   - **Bi-encoder**: embeds query and documents *separately*, compares vectors — cheap, scales to millions of docs, coarser relevance
   - **Cross-encoder**: feeds the *(query, document) pair together* through the model with cross-attention — much more accurate relevance, but far slower

2. **Two-Stage Retrieval Pattern**:
   - Cross-encoders are too slow to run over a whole corpus, so they are used as a **reranker** over a small candidate set, not as the first-stage retriever
   - Retrieve wide and cheap (bi-encoder), then rerank narrow and accurate (cross-encoder) — the canonical production RAG pattern

3. **Why Reranking Helps**:
   - Pure embedding similarity can rank a document highly for sharing surface vocabulary with the query while not answering it
   - Joint scoring lets the reranker demote those lexical "traps" and promote genuinely relevant documents

4. **Practical Notes**:
   - `CrossEncoder.predict()` takes a list of `[query, passage]` pairs and returns one relevance score per pair (higher = better) — sort descending
   - Retrieve more candidates than you keep (e.g., retrieve 6, keep 3) to give the reranker room to reorder
   - `sentence-transformers` (already a dependency of the embeddings stack) provides `CrossEncoder`, so no new heavy dependency is needed

---

This session provided hands-on experience upgrading a naive single-pass retriever into a two-stage retrieve-then-rerank pipeline — combining a fast bi-encoder for wide recall with an accurate cross-encoder for precise reordering, the canonical pattern for production-grade RAG retrieval quality.