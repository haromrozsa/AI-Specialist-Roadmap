from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ===========================================
# 1. DOCUMENT LOADER
# ===========================================

def load_documents(source_path: str, file_type: str = "txt"):
    """
    Load documents from a file or directory.

    Args:
        source_path: Path to file or directory
        file_type: Type of files to load ('txt', 'pdf', 'directory')

    Returns:
        List of Document objects
    """
    if file_type == "txt":
        loader = TextLoader(source_path, encoding="utf-8")
    elif file_type == "pdf":
        loader = PyPDFLoader(source_path)
    elif file_type == "directory":
        loader = DirectoryLoader(
            path=source_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document(s)")
    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks for better retrieval.

    Args:
        documents: List of Document objects
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks for context preservation

    Returns:
        List of Document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ===========================================
# 2. EMBEDDINGS (HuggingFace)
# ===========================================

def create_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create HuggingFace embeddings model.

    Args:
        model_name: Name of the HuggingFace embedding model

    Returns:
        HuggingFaceEmbeddings instance
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"✅ Created embeddings model: {model_name}")
    return embeddings


# ===========================================
# 3. VECTOR STORE (FAISS)
# ===========================================

def create_vector_store(chunks, embeddings, persist_path: str = None):
    """
    Create a FAISS vector store from document chunks.

    Args:
        chunks: List of Document chunks
        embeddings: Embedding model
        persist_path: Optional path to save the vector store

    Returns:
        FAISS vector store instance
    """
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    if persist_path:
        vector_store.save_local(persist_path)
        print(f"✅ Vector store saved to: {persist_path}")

    print(f"✅ Created vector store with {len(chunks)} vectors")
    return vector_store


def load_vector_store(persist_path: str, embeddings):
    """
    Load an existing FAISS vector store.

    Args:
        persist_path: Path to the saved vector store
        embeddings: Embedding model (must match the one used to create)

    Returns:
        FAISS vector store instance
    """
    vector_store = FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ Loaded vector store from: {persist_path}")
    return vector_store


# ===========================================
# 4. LLM SETUP (HuggingFace)
# ===========================================

def create_llm(model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
    """
    Create a HuggingFace LLM pipeline.

    Args:
        model_id: HuggingFace model ID

    Returns:
        HuggingFacePipeline LLM instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,  # Lower temperature for more factual responses
        top_k=50,
        do_sample=True,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print(f"✅ Created LLM: {model_id}")
    return llm


# ===========================================
# 5. RETRIEVAL QA CHAIN
# ===========================================

def create_rag_chain(vector_store, llm, k: int = 4):
    """
    Create a RAG chain for question answering.

    Args:
        vector_store: FAISS vector store
        llm: Language model
        k: Number of documents to retrieve

    Returns:
        RAG chain
    """
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # RAG prompt template - instructs the model to ONLY use provided context
    template = """You are a helpful assistant that answers questions based ONLY on the provided context.
If the context doesn't contain relevant information to answer the question, say "I don't have enough information in the provided documents to answer this question."
Do NOT make up information or use knowledge outside the provided context.

Context:
{context}

Question: {question}

Answer based only on the context above:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the RAG chain using LCEL (LangChain Expression Language)
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print("✅ Created RAG chain")
    return rag_chain, retriever


def ask_question(rag_chain, retriever, question: str, show_sources: bool = True):
    """
    Ask a question and get an answer grounded in the documents.

    Args:
        rag_chain: The RAG chain
        retriever: The retriever (for showing sources)
        question: The question to ask
        show_sources: Whether to show source documents

    Returns:
        Answer string
    """
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print('=' * 60)

    # Get the answer
    answer = rag_chain.invoke(question)

    print(f"\nAnswer: {answer}")

    # Optionally show the source documents used
    if show_sources:
        source_docs = retriever.invoke(question)
        print(f"\n📚 Sources ({len(source_docs)} documents retrieved):")
        for i, doc in enumerate(source_docs, 1):
            print(f"\n--- Source {i} ---")
            print(f"Content: {doc.page_content[:200]}...")
            if doc.metadata:
                print(f"Metadata: {doc.metadata}")

    return answer


# ===========================================
# MAIN EXECUTION
# ===========================================

def main():
    # Configuration
    DOCUMENTS_PATH = "./documents"  # Change to your documents path
    VECTOR_STORE_PATH = "./faiss_index"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"

    # Step 1: Load documents
    print("\n📄 Loading documents...")
    # For a single text file:
    # documents = load_documents("your_document.txt", file_type="txt")
    # For a directory of text files:
    documents = load_documents(DOCUMENTS_PATH, file_type="directory")

    # Step 2: Split into chunks
    print("\n✂️ Splitting documents...")
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)

    # Step 3: Create embeddings
    print("\n🔢 Creating embeddings...")
    embeddings = create_embeddings(EMBEDDING_MODEL)

    # Step 4: Create vector store
    print("\n📊 Creating vector store...")
    vector_store = create_vector_store(chunks, embeddings, VECTOR_STORE_PATH)

    # Step 5: Create LLM
    print("\n🤖 Loading LLM...")
    llm = create_llm(LLM_MODEL)

    # Step 6: Create RAG chain
    print("\n🔗 Creating RAG chain...")
    rag_chain, retriever = create_rag_chain(vector_store, llm, k=4)

    # Step 7: Ask questions!
    print("\n" + "=" * 60)
    print("RAG System Ready! Ask questions based on your documents.")
    print("=" * 60)

    # Example questions
    questions = [
        "What is the main topic of the documents?",
        "Can you summarize the key points?",
    ]

    for question in questions:
        ask_question(rag_chain, retriever, question, show_sources=True)


if __name__ == "__main__":
    main()