from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# Quick setup with in-memory documents
def quick_rag_demo():
    # Sample documents (in-memory)
    from langchain_core.documents import Document

    documents = [
        Document(page_content="Python was created by Guido van Rossum in 1991. It emphasizes code readability."),
        Document(page_content="LangChain is a framework for developing applications powered by language models."),
        Document(page_content="RAG combines retrieval systems with generative models to produce grounded responses."),
        Document(page_content="FAISS is a library for efficient similarity search developed by Facebook AI Research."),
    ]

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(documents, embeddings)

    # Retrieve relevant documents
    query = "What is LangChain?"
    relevant_docs = vector_store.similarity_search(query, k=2)

    print(f"Query: {query}\n")
    print("Retrieved Documents:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"{i}. {doc.page_content}")

    # The context can now be passed to your LLM for grounded answer generation
    context = "\n".join([doc.page_content for doc in relevant_docs])
    print(f"\nContext for LLM:\n{context}")


if __name__ == "__main__":
    quick_rag_demo()