#!/usr/bin/env python3
"""
Embedding API client examples for Jetson LLM API.

These examples demonstrate how to use the /v1/embeddings endpoint
to generate vector embeddings for text inputs.
"""

from openai import OpenAI
import numpy as np

# Initialize OpenAI client pointing to local API
client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="change-me-to-a-secure-key"
)


def example_single_embedding():
    """Generate embedding for a single text string."""
    print("=== Example 1: Single Text Embedding ===")

    text = "The quick brown fox jumps over the lazy dog"

    response = client.embeddings.create(
        model="qwen3-embedding-8b",
        input=text
    )

    # Extract the embedding vector
    embedding = response.data[0].embedding

    print(f"Input text: {text}")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Token usage: {response.usage.total_tokens} tokens")
    print()

    return embedding


def example_batch_embeddings():
    """Generate embeddings for multiple texts in one request."""
    print("=== Example 2: Batch Embeddings ===")

    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing focuses on understanding text",
        "Computer vision enables machines to interpret images"
    ]

    response = client.embeddings.create(
        model="qwen3-embedding-8b",
        input=texts
    )

    print(f"Generated {len(response.data)} embeddings")
    for i, embedding_obj in enumerate(response.data):
        print(f"  Text {i+1}: {len(embedding_obj.embedding)} dimensions")
    print(f"Total token usage: {response.usage.total_tokens} tokens")
    print()

    # Return all embeddings
    return [emb.embedding for emb in response.data]


def example_similarity_search():
    """Use embeddings to find similar texts via cosine similarity."""
    print("=== Example 3: Similarity Search ===")

    # Query and candidate documents
    query = "How does AI learn from data?"
    documents = [
        "Machine learning algorithms learn patterns from training data",
        "The weather forecast predicts rain tomorrow",
        "Deep neural networks require large datasets for training",
        "Pizza is a popular Italian dish",
        "Supervised learning uses labeled data to train models"
    ]

    # Generate embeddings for query and documents
    all_texts = [query] + documents
    response = client.embeddings.create(
        model="qwen3-embedding-8b",
        input=all_texts
    )

    # Extract embeddings
    query_embedding = np.array(response.data[0].embedding)
    doc_embeddings = np.array([emb.embedding for emb in response.data[1:]])

    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]

    # Sort documents by similarity
    ranked_docs = sorted(
        zip(documents, similarities),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"Query: '{query}'")
    print("\nRanked results:")
    for i, (doc, score) in enumerate(ranked_docs, 1):
        print(f"{i}. [Score: {score:.4f}] {doc}")
    print()


def example_document_clustering():
    """Cluster documents using their embeddings."""
    print("=== Example 4: Document Clustering ===")

    documents = [
        "Python is a versatile programming language",
        "JavaScript is widely used for web development",
        "The Eiffel Tower is located in Paris",
        "Machine learning models require training data",
        "The Great Wall of China is a historic landmark",
        "Neural networks are inspired by biological neurons",
        "Rome is known for its ancient architecture",
        "TypeScript adds static typing to JavaScript"
    ]

    # Generate embeddings
    response = client.embeddings.create(
        model="qwen3-embedding-8b",
        input=documents
    )

    embeddings = np.array([emb.embedding for emb in response.data])

    # Simple clustering using K-means (requires sklearn)
    try:
        from sklearn.cluster import KMeans

        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        print(f"Clustered {len(documents)} documents into {n_clusters} groups:\n")
        for cluster_id in range(n_clusters):
            print(f"Cluster {cluster_id + 1}:")
            cluster_docs = [doc for doc, label in zip(documents, labels) if label == cluster_id]
            for doc in cluster_docs:
                print(f"  - {doc}")
            print()

    except ImportError:
        print("Install scikit-learn to run clustering: pip install scikit-learn")
        print(f"Generated embeddings for {len(documents)} documents")
        print(f"Embedding shape: {embeddings.shape}")


def example_semantic_search_with_threshold():
    """Filter results by similarity threshold."""
    print("=== Example 5: Semantic Search with Threshold ===")

    query = "renewable energy sources"
    documents = [
        "Solar panels convert sunlight into electricity",
        "Wind turbines generate power from wind energy",
        "The stock market experienced volatility today",
        "Hydroelectric dams produce renewable energy",
        "Cats are popular household pets",
        "Geothermal energy harnesses heat from the Earth"
    ]

    # Generate embeddings
    all_texts = [query] + documents
    response = client.embeddings.create(
        model="qwen3-embedding-8b",
        input=all_texts
    )

    query_embedding = np.array(response.data[0].embedding)
    doc_embeddings = np.array([emb.embedding for emb in response.data[1:]])

    # Calculate similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]

    # Filter by threshold
    threshold = 0.5
    relevant_docs = [
        (doc, score)
        for doc, score in zip(documents, similarities)
        if score >= threshold
    ]

    print(f"Query: '{query}'")
    print(f"Similarity threshold: {threshold}")
    print(f"\nRelevant documents ({len(relevant_docs)} found):")
    for doc, score in sorted(relevant_docs, key=lambda x: x[1], reverse=True):
        print(f"  [Score: {score:.4f}] {doc}")
    print()


if __name__ == "__main__":
    print("Jetson LLM API - Embedding Examples\n")
    print("Make sure the API and embedding backend are running:")
    print("  - API: http://localhost:9000")
    print("  - Embedding backend: http://localhost:8085")
    print()

    try:
        # Run examples
        example_single_embedding()
        example_batch_embeddings()
        example_similarity_search()
        example_document_clustering()
        example_semantic_search_with_threshold()

        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. The jetson-api service is running")
        print("2. The llama-qwen3-embedding service is running")
        print("3. Your API key matches the one in .env")
