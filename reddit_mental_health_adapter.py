%%writefile reddit_mental_health_adapter.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit Mental Health Dataset Adapter - DAR-Compatible Version
Converts Reddit posts to BEIR-style JSONL format for DAR Router experiments
Uses TF-IDF vectorization for 100x faster qrels generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re
from datasets import load_dataset
from tqdm import tqdm
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text: str) -> str:
    """Clean Reddit text (remove markdown, fix formatting)"""
    if not isinstance(text, str):
        return ""
    
    # Remove Reddit markdown
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'~~([^~]+)~~', r'\1', text)  # Strikethrough
    
    # Fix common issues
    text = re.sub(r'\n+', ' ', text)  # Multiple newlines
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces
    text = text.strip()
    
    return text

def extract_query_from_post(post: Dict) -> str:
    """Extract query from Reddit post (title + snippet of text)"""
    title = clean_text(post.get('title', ''))
    text = clean_text(post.get('selftext', ''))
    
    # Combine title and first part of text
    if text:
        # Take first 300 chars of text for query
        text_snippet = text[:300] + "..." if len(text) > 300 else text
        query = f"{title} {text_snippet}".strip()
    else:
        query = title
    
    return query

def create_document_pool(posts: List[Dict], min_doc_length: int = 50) -> List[Dict]:
    """
    Create document pool from Reddit posts.
    Each post's full text becomes a document that can be retrieved.
    """
    documents = []
    doc_id = 0
    
    for post in tqdm(posts, desc="Creating document pool"):
        # Use full text as document
        doc_text = clean_text(
            (post.get('title', '') or '') + ' ' + (post.get('selftext', '') or '')
        )
        
        # Skip if too short
        if len(doc_text) < min_doc_length:
            continue
        
        documents.append({
            'id': f'd{doc_id}',  # BEIR format uses string IDs
            'text': doc_text,
            'metadata': {
                'source_post_id': post.get('post_id', f"post_{doc_id}"),
                'subreddit': post.get('subreddit', 'unknown')
            }
        })
        doc_id += 1
    
    return documents

def create_relevance_judgments_FAST(
    queries: List[Dict],
    documents: List[Dict],
    max_relevant_per_query: int = 10,
    min_similarity: float = 0.05
) -> List[Dict]:
    """
    Create relevance judgments using TF-IDF + cosine similarity.
    THIS IS 100x FASTER than nested loop approach!
    
    Returns BEIR-style qrels: [{'query-id': ..., 'corpus-id': ..., 'score': ...}]
    """
    print(f"\n{'='*70}")
    print("FAST QRELS GENERATION (TF-IDF + Cosine Similarity)")
    print(f"{'='*70}")
    
    np.random.seed(42)
    
    # Extract texts
    print("[1/4] Extracting query and document texts...")
    query_texts = [q['text'] for q in queries]
    doc_texts = [d['text'] for d in documents]
    
    print(f"  ✓ Queries: {len(query_texts):,}")
    print(f"  ✓ Documents: {len(doc_texts):,}")
    
    # Build TF-IDF vectorizer
    print("\n[2/4] Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Fit on all texts
    all_texts = query_texts + doc_texts
    vectorizer.fit(all_texts)
    print(f"  ✓ Vocabulary size: {len(vectorizer.vocabulary_):,}")
    
    # Transform queries and documents
    print("\n[3/4] Vectorizing queries and documents...")
    query_vectors = vectorizer.transform(query_texts)
    doc_vectors = vectorizer.transform(doc_texts)
    print(f"  ✓ Query vectors: {query_vectors.shape}")
    print(f"  ✓ Document vectors: {doc_vectors.shape}")
    
    # Compute cosine similarity matrix
    print("\n[4/4] Computing similarity matrix...")
    print(f"  Matrix size: {len(queries):,} × {len(documents):,} = {len(queries) * len(documents):,} cells")
    
    # Compute in batches to avoid memory issues
    batch_size = 500
    qrels = []
    
    for batch_start in tqdm(range(0, len(queries), batch_size), desc="Computing similarities"):
        batch_end = min(batch_start + batch_size, len(queries))
        
        # Get similarity for this batch
        batch_similarities = cosine_similarity(
            query_vectors[batch_start:batch_end],
            doc_vectors
        )
        
        # Process each query in batch
        for i, global_idx in enumerate(range(batch_start, batch_end)):
            query = queries[global_idx]
            query_id = query['id']
            
            # Get top-k most similar documents
            similarities = batch_similarities[i]
            top_k_indices = np.argsort(similarities)[-max_relevant_per_query * 2:][::-1]
            
            # Assign relevance grades based on similarity
            for ans_idx in top_k_indices[:max_relevant_per_query]:
                sim_score = similarities[ans_idx]
                
                # Skip if similarity too low
                if sim_score < min_similarity:
                    continue
                
                # Assign relevance grade (BEIR uses continuous scores)
                # We convert similarity to relevance score
                if sim_score > 0.3:
                    score = 3  # Highly relevant
                elif sim_score > 0.15:
                    score = 2  # Relevant
                elif sim_score > 0.08:
                    score = 1  # Marginally relevant
                else:
                    continue
                
                qrels.append({
                    'query-id': query_id,
                    'corpus-id': documents[ans_idx]['id'],
                    'score': score
                })
    
    print(f"\n  ✓ Generated {len(qrels):,} relevance judgments")
    print(f"  ✓ Avg per query: {len(qrels)/len(queries):.1f}")
    print(f"{'='*70}\n")
    
    return qrels

def load_and_convert_reddit_dataset(
    output_dir: str = "paper_experiments/data/reddit_mental_health",
    num_queries: int = 2000,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Reddit mental health dataset and convert to BEIR-style JSONL format.
    OPTIMIZED VERSION - uses TF-IDF for fast qrels generation
    
    Output files (BEIR format):
        - corpus.jsonl: [{'id': 'd0', 'text': '...'}]
        - queries.jsonl: [{'id': 'q0', 'text': '...'}]
        - qrels.jsonl: [{'query-id': 'q0', 'corpus-id': 'd0', 'score': 3}]
    
    Returns:
        queries_df, docs_df, qrels_df
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("REDDIT MENTAL HEALTH DATASET ADAPTER (BEIR-Compatible)")
    print("=" * 70)
    
    # Load dataset from HuggingFace
    print("\n[1/6] Loading dataset from HuggingFace...")
    try:
        dataset = load_dataset("solomonk/reddit_mental_health_posts", split="train")
        print(f"  ✓ Loaded {len(dataset):,} posts")
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        print(f"  💡 Install with: pip install datasets")
        raise
    
    # Convert to list of dicts
    posts = []
    for i, item in enumerate(dataset):
        posts.append({
            'post_id': f"reddit_{i}",
            'title': item.get('title', ''),
            'selftext': item.get('selftext') or item.get('body', ''),
            'subreddit': item.get('subreddit', 'mentalhealth')
        })
    
    print(f"  ✓ Converted to {len(posts):,} posts")
    
    # Sample queries
    print(f"\n[2/6] Sampling {num_queries:,} queries...")
    if len(posts) > num_queries:
        sampled_indices = np.random.choice(len(posts), size=num_queries, replace=False)
        sampled_posts = [posts[i] for i in sampled_indices]
    else:
        sampled_posts = posts
        num_queries = len(posts)
    
    print(f"  ✓ Selected {len(sampled_posts):,} posts as queries")
    
    # Create queries
    print("\n[3/6] Creating queries...")
    queries = []
    for i, post in enumerate(tqdm(sampled_posts, desc="Processing queries")):
        query_text = extract_query_from_post(post)
        
        # Skip empty or very short queries
        if len(query_text) < 20:
            continue
        
        queries.append({
            'id': f'q{i}',  # BEIR format: string IDs
            'text': query_text,
            'metadata': {
                'subreddit': post.get('subreddit', 'mentalhealth'),
                'source_post_id': post.get('post_id')
            }
        })
    
    print(f"  ✓ Created {len(queries):,} queries")
    
    # Create document pool from ALL posts (not just sampled ones)
    print("\n[4/6] Creating document pool from all posts...")
    documents = create_document_pool(posts, min_doc_length=50)
    print(f"  ✓ Created {len(documents):,} documents")
    
    # Create relevance judgments (FAST VERSION!)
    print("\n[5/6] Creating relevance judgments (FAST)...")
    qrels = create_relevance_judgments_FAST(
        queries, 
        documents, 
        max_relevant_per_query=10,
        min_similarity=0.05
    )
    print(f"  ✓ Created {len(qrels):,} relevance judgments")
    
    # Convert to DataFrames for inspection
    queries_df = pd.DataFrame(queries)
    docs_df = pd.DataFrame(documents)
    qrels_df = pd.DataFrame(qrels)
    
    # Save to BEIR-style JSONL files
    print("\n[6/6] Saving BEIR-style JSONL files...")
    
    # Save corpus.jsonl (documents)
    with open(output_path / "corpus.jsonl", 'w') as f:
        for doc in documents:
            # Only include id and text in JSONL (metadata is optional)
            f.write(json.dumps({'id': doc['id'], 'text': doc['text']}) + '\n')
    
    # Save queries.jsonl
    with open(output_path / "queries.jsonl", 'w') as f:
        for query in queries:
            f.write(json.dumps({'id': query['id'], 'text': query['text']}) + '\n')
    
    # Save qrels.jsonl
    with open(output_path / "qrels.jsonl", 'w') as f:
        for qrel in qrels:
            f.write(json.dumps(qrel) + '\n')
    
    # Also save as parquet for convenience
    queries_df.to_parquet(output_path / "queries.parquet", index=False)
    docs_df.to_parquet(output_path / "docs.parquet", index=False)
    qrels_df.to_parquet(output_path / "qrels.parquet", index=False)
    
    # Save metadata
    metadata = {
        'dataset': 'reddit_mental_health',
        'source': 'solomonk/reddit_mental_health_posts',
        'format': 'BEIR-compatible JSONL',
        'num_queries': len(queries),
        'num_docs': len(documents),
        'num_qrels': len(qrels),
        'seed': seed,
        'created': pd.Timestamp.now().isoformat(),
        'method': 'tfidf_cosine_similarity',
        'files': {
            'corpus': 'corpus.jsonl',
            'queries': 'queries.jsonl',
            'qrels': 'qrels.jsonl'
        }
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ DATASET CONVERSION COMPLETE")
    print("=" * 70)
    print(f"\nStatistics:")
    print(f"  Queries:  {len(queries):,}")
    print(f"  Docs:     {len(documents):,}")
    print(f"  Qrels:    {len(qrels):,}")
    print(f"  Avg qrels per query: {len(qrels)/len(queries):.1f}")
    
    # Distribution analysis
    rel_counts = qrels_df['score'].value_counts().sort_index()
    print(f"\nRelevance distribution:")
    for score, count in rel_counts.items():
        print(f"  Score {score}: {count:,} ({count/len(qrels)*100:.1f}%)")
    
    print(f"\nSaved to: {output_path}/")
    print(f"  📄 corpus.jsonl  - {len(documents):,} documents")
    print(f"  📄 queries.jsonl - {len(queries):,} queries")
    print(f"  📄 qrels.jsonl   - {len(qrels):,} judgments")
    print(f"  📄 metadata.json - dataset info")
    print("\n💡 Use with DAR:")
    print(f"   python dar_router_main.py \\")
    print(f"       --data_root {output_path} \\")
    print(f"       --workdir paper_experiments/runs/reddit \\")
    print(f"       --device auto")
    print("=" * 70)
    
    return queries_df, docs_df, qrels_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reddit Mental Health Dataset Adapter (BEIR-Compatible)")
    parser.add_argument("--output_dir", type=str, default="paper_experiments/data/reddit_mental_health",
                       help="Output directory for converted dataset")
    parser.add_argument("--num_queries", type=int, default=2000,
                       help="Number of queries to sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Run the adapter
    queries_df, docs_df, qrels_df = load_and_convert_reddit_dataset(
        output_dir=args.output_dir,
        num_queries=args.num_queries,
        seed=args.seed
    )
    
    # Print samples
    print("\n📋 SAMPLE QUERIES:")
    print(queries_df.head(3)[['id', 'text']].to_string(index=False))
    
    print("\n📋 SAMPLE DOCS:")
    print(docs_df.head(3)[['id', 'text']].to_string(index=False))
    
    print("\n📋 SAMPLE QRELS:")
    print(qrels_df.head(10).to_string(index=False))
