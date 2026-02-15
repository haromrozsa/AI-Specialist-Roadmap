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