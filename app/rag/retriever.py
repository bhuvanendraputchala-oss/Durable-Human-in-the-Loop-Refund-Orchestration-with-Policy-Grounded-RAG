from .vectorstore import get_vectorstore


def retrieve_similarity(query: str, k: int = 4):
    """Standard cosine-similarity retrieval."""
    vectordb = get_vectorstore()
    return vectordb.similarity_search(query, k=k)


def retrieve_mmr(query: str, k: int = 4):
    """Max-marginal-relevance retrieval for diverse result sets."""
    vectordb = get_vectorstore()
    return vectordb.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)

def retrieve(query: str, k: int = 6):
    return retrieve_similarity(query, k=k)
