"""
Reranking RAG: bi-encoder retrieve -> cross-encoder rerank
==========================================================

The base RAG demo (``rag_system.py``) retrieves with a single **bi-encoder**:
the query and every document are embedded *independently*, and we keep the
nearest vectors. That is fast and scales to millions of docs, but it is a coarse
relevance signal -- a document can sit close in embedding space because it shares
surface vocabulary with the query while not actually answering it.

The fix is a **two-stage retriever**:

1. **Retrieve (bi-encoder / FAISS)** -- cheaply pull a wider candidate set
   (here top-6) from the whole corpus.
2. **Rerank (cross-encoder)** -- for each candidate, feed the *(query, document)
   pair together* into a cross-encoder that outputs a single relevance score.
   Scoring the pair jointly (with cross-attention) is far more accurate than
   comparing two independent embeddings, so we re-sort and keep only the best
   few (here top-3).

Cross-encoders are too slow to run over an entire corpus, which is exactly why
they are used as a *reranker* over a small candidate set rather than as the
first-stage retriever. This is the canonical production RAG retrieval pattern.

Self-contained and CPU-only: in-memory documents, no external files, no LLM
download. The example corpus is crafted so pure embedding similarity ranks a
couple of lexical "trap" documents high, and the cross-encoder demotes them.
"""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"     # bi-encoder (retrieval)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"        # cross-encoder (rerank)
RETRIEVE_K = 6   # wide candidate set pulled by FAISS
TOP_N = 3        # final documents kept after reranking

QUERY = "How can I reduce the memory footprint of a large language model for deployment?"

# The corpus mixes genuinely relevant model-compression docs with "trap" docs
# that share vocabulary with the query ("memory", "deployment", "large ...
# model") but do not answer it. A good reranker should surface the compression
# docs and demote the traps.
DOCUMENTS = [
    Document(page_content="Quantization shrinks a model by storing its weights in lower-precision formats such as int8 instead of float32, cutting memory use during inference and deployment.",
             metadata={"title": "Quantization", "relevant": True}),
    Document(page_content="Knowledge distillation trains a smaller student model to imitate a larger teacher, producing a compact model with a much smaller footprint.",
             metadata={"title": "Distillation", "relevant": True}),
    Document(page_content="Pruning removes redundant or low-importance weights from a neural network, reducing its size and the memory required to run it.",
             metadata={"title": "Pruning", "relevant": True}),
    Document(page_content="Large language models have grown to hundreds of billions of parameters, demanding enormous compute and data to train from scratch.",
             metadata={"title": "LLM scale (trap: 'large ... model')", "relevant": False}),
    Document(page_content="Human memory is commonly divided into short-term and long-term systems, a distinction studied extensively in cognitive psychology.",
             metadata={"title": "Human memory (trap: 'memory')", "relevant": False}),
    Document(page_content="Deploying web applications reliably usually involves load balancing, horizontal scaling, and rolling releases across servers.",
             metadata={"title": "Web deployment (trap: 'deploying')", "relevant": False}),
    Document(page_content="GPU memory bandwidth is a major factor in transformer inference latency when serving requests at scale.",
             metadata={"title": "GPU bandwidth (trap: 'memory')", "relevant": False}),
    Document(page_content="The transformer architecture relies on self-attention to model relationships between tokens in a sequence.",
             metadata={"title": "Transformer background", "relevant": False}),
]


def build_vector_store():
    """Embed the corpus with the bi-encoder and index it in FAISS."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = FAISS.from_documents(DOCUMENTS, embeddings)
    print(f"✅ Indexed {len(DOCUMENTS)} documents with {EMBEDDING_MODEL}")
    return store


def retrieve(store, query, k):
    """Stage 1: FAISS nearest-neighbour search.

    Returns a list of (Document, distance) sorted by ascending distance
    (lower distance = more similar).
    """
    results = store.similarity_search_with_score(query, k=k)
    print(f"\n{'=' * 70}\nSTAGE 1 — Bi-encoder retrieval (FAISS, top {k})\n{'=' * 70}")
    print("(distance: lower = closer in embedding space)\n")
    for rank, (doc, distance) in enumerate(results, 1):
        print(f"  {rank}. [dist {distance:.3f}] {doc.metadata['title']}")
    return results


def rerank(query, retrieved, top_n):
    """Stage 2: cross-encoder reranking of the retrieved candidates.

    Scores each (query, document) pair jointly and keeps the top ``top_n``.
    Returns a list of (Document, rerank_score, original_vector_rank).
    """
    reranker = CrossEncoder(RERANKER_MODEL)
    candidates = [doc for doc, _ in retrieved]

    # Cross-encoder input is a list of [query, passage] pairs; output is one
    # relevance score per pair (higher = more relevant).
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    # Remember each candidate's rank from stage 1 so we can show the movement.
    original_rank = {id(doc): i + 1 for i, doc in enumerate(candidates)}

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    print(f"\n{'=' * 70}\nSTAGE 2 — Cross-encoder rerank ({RERANKER_MODEL}, top {top_n})\n{'=' * 70}")
    print("(score: higher = more relevant; ▲/▼ shows move vs. stage-1 rank)\n")
    top = []
    for new_rank, (doc, score) in enumerate(ranked[:top_n], 1):
        old = original_rank[id(doc)]
        move = "▲" if old > new_rank else ("▼" if old < new_rank else "=")
        print(f"  {new_rank}. [score {score:+.2f}] {move} was #{old}  {doc.metadata['title']}")
        top.append((doc, score, old))
    return top


def main():
    print(f"Query: {QUERY}")
    store = build_vector_store()

    retrieved = retrieve(store, QUERY, RETRIEVE_K)
    top = rerank(QUERY, retrieved, TOP_N)

    # Assemble the reranked top-N as grounded context, ready to hand to an LLM
    # (the generation step from rag_system.py). Keeping the demo retrieval-only
    # means it stays fast and CPU-only.
    print(f"\n{'=' * 70}\nGrounded context for the LLM (reranked top {TOP_N})\n{'=' * 70}")
    context = "\n\n".join(doc.page_content for doc, _, _ in top)
    print(context)

    kept_relevant = sum(doc.metadata["relevant"] for doc, _, _ in top)
    print(f"\n✅ {kept_relevant}/{TOP_N} reranked documents are the genuinely relevant "
          f"model-compression docs — the lexical 'trap' documents were demoted.")


if __name__ == "__main__":
    main()
