"""Tests for vectorization functions."""
import pytest
import numpy as np
from src.features.vectorization import (
    compute_tf, compute_idf, compute_tfidf,
    compute_tf_vectorized, compute_idf_vectorized, compute_tfidf_vectorized
)


@pytest.fixture
def sample_tokens():
    """Create sample tokens for testing."""
    return ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']


@pytest.fixture
def sample_corpus():
    """Create sample corpus for testing."""
    return [
        ['the', 'quick', 'brown', 'fox'],
        ['the', 'lazy', 'dog'],
        ['the', 'quick', 'dog']
    ]


def test_compute_tf(sample_tokens):
    """Test compute_tf function."""
    tf_dict = compute_tf(sample_tokens)
    
    # Check that we get a dictionary
    assert isinstance(tf_dict, dict)
    
    # Check that frequencies sum to 1
    assert abs(sum(tf_dict.values()) - 1.0) < 1e-10
    
    # Check specific values
    assert tf_dict['the'] == 2/9  # 'the' appears twice in 9 tokens
    assert tf_dict['quick'] == 1/9  # 'quick' appears once in 9 tokens
    
    # Check all tokens are present
    for token in set(sample_tokens):
        assert token in tf_dict


def test_compute_tf_vectorized(sample_tokens):
    """Test compute_tf_vectorized function."""
    tf_dict = compute_tf_vectorized(sample_tokens)
    
    # Check that we get a dictionary
    assert isinstance(tf_dict, dict)
    
    # Check that frequencies sum to 1
    assert abs(sum(tf_dict.values()) - 1.0) < 1e-10
    
    # Check specific values
    assert tf_dict['the'] == 2/9  # 'the' appears twice in 9 tokens
    assert tf_dict['quick'] == 1/9  # 'quick' appears once in 9 tokens
    
    # Check all tokens are present
    for token in set(sample_tokens):
        assert token in tf_dict
    
    # Compare with non-vectorized version
    non_vectorized = compute_tf(sample_tokens)
    for token in non_vectorized:
        assert abs(tf_dict[token] - non_vectorized[token]) < 1e-10


def test_compute_idf(sample_corpus):
    """Test compute_idf function."""
    idf_dict = compute_idf(sample_corpus)
    
    # Check that we get a dictionary
    assert isinstance(idf_dict, dict)
    
    # Check specific values
    assert idf_dict['the'] == 0.0  # 'the' appears in all documents
    assert idf_dict['brown'] > 0  # 'brown' appears in only one document
    
    # Term appearing in all docs should have lowest IDF
    assert all(idf_dict['the'] <= value for value in idf_dict.values())
    
    # Check all unique terms in corpus are present
    all_terms = set()
    for doc in sample_corpus:
        all_terms.update(doc)
    assert set(idf_dict.keys()) == all_terms


def test_compute_idf_vectorized(sample_corpus):
    """Test compute_idf_vectorized function."""
    idf_dict = compute_idf_vectorized(sample_corpus)
    
    # Check that we get a dictionary
    assert isinstance(idf_dict, dict)
    
    # Check specific values
    assert idf_dict['the'] == 0.0  # 'the' appears in all documents
    assert idf_dict['brown'] > 0  # 'brown' appears in only one document
    
    # Term appearing in all docs should have lowest IDF
    assert all(idf_dict['the'] <= value for value in idf_dict.values())
    
    # Check all unique terms in corpus are present
    all_terms = set()
    for doc in sample_corpus:
        all_terms.update(doc)
    assert set(idf_dict.keys()) == all_terms
    
    # Compare with non-vectorized version
    non_vectorized = compute_idf(sample_corpus)
    for token in non_vectorized:
        assert abs(idf_dict[token] - non_vectorized[token]) < 1e-10


def test_compute_tfidf(sample_corpus):
    """Test compute_tfidf function."""
    # Prepare inputs
    tf_dicts = [compute_tf(doc) for doc in sample_corpus]
    idf_dict = compute_idf(sample_corpus)
    
    # Compute TF-IDF
    tfidf_docs = compute_tfidf(tf_dicts, idf_dict)
    
    # Check that we get a list of dictionaries
    assert isinstance(tfidf_docs, list)
    assert len(tfidf_docs) == len(sample_corpus)
    assert all(isinstance(doc, dict) for doc in tfidf_docs)
    
    # Check TF-IDF calculation correctness
    for i, (tf_dict, tfidf_dict) in enumerate(zip(tf_dicts, tfidf_docs)):
        for term in tf_dict:
            expected = tf_dict[term] * idf_dict[term]
            assert abs(tfidf_dict[term] - expected) < 1e-10
    
    # Term with 0 IDF should have 0 TF-IDF
    for doc in tfidf_docs:
        assert doc['the'] == 0.0


def test_compute_tfidf_vectorized(sample_corpus):
    """Test compute_tfidf_vectorized function."""
    # Compute TF-IDF using vectorized function
    tfidf_docs, idf_dict = compute_tfidf_vectorized(sample_corpus)
    
    # Check that we get expected types
    assert isinstance(tfidf_docs, list)
    assert len(tfidf_docs) == len(sample_corpus)
    assert all(isinstance(doc, dict) for doc in tfidf_docs)
    assert isinstance(idf_dict, dict)
    
    # Check TF-IDF calculation correctness using individual components
    tf_dicts = [compute_tf_vectorized(doc) for doc in sample_corpus]
    for i, (tf_dict, tfidf_dict) in enumerate(zip(tf_dicts, tfidf_docs)):
        for term in tf_dict:
            if term in idf_dict:  # Skip terms not in idf_dict
                expected = tf_dict[term] * idf_dict[term]
                assert abs(tfidf_dict.get(term, 0) - expected) < 1e-10
    
    # Term with 0 IDF should have 0 TF-IDF
    for doc in tfidf_docs:
        assert doc.get('the', None) == 0.0


def test_vectorized_matches_non_vectorized(sample_corpus):
    """Test that vectorized implementations match non-vectorized ones."""
    # Check TF implementations
    for doc in sample_corpus:
        tf1 = compute_tf(doc)
        tf2 = compute_tf_vectorized(doc)
        assert set(tf1.keys()) == set(tf2.keys())
        for term in tf1:
            assert abs(tf1[term] - tf2[term]) < 1e-10
    
    # Check IDF implementations
    idf1 = compute_idf(sample_corpus)
    idf2 = compute_idf_vectorized(sample_corpus)
    assert set(idf1.keys()) == set(idf2.keys())
    for term in idf1:
        assert abs(idf1[term] - idf2[term]) < 1e-10
    
    # Check TF-IDF implementations
    tf_dicts = [compute_tf(doc) for doc in sample_corpus]
    tfidf1 = compute_tfidf(tf_dicts, idf1)
    tfidf2, _ = compute_tfidf_vectorized(sample_corpus)
    
    # The orders might be different, so we check document by document
    for doc1, doc2 in zip(tfidf1, tfidf2):
        assert set(doc1.keys()) == set(doc2.keys())
        for term in doc1:
            assert abs(doc1[term] - doc2[term]) < 1e-10


def test_empty_inputs():
    """Test handling of empty inputs."""
    # Empty tokens list
    assert compute_tf([]) == {}
    assert compute_tf_vectorized([]) == {}
    
    # Empty corpus
    assert compute_idf([]) == {}
    assert compute_idf_vectorized([]) == {}
    
    # Empty TF dict list
    assert compute_tfidf([], {}) == []
    
    # Empty corpus for vectorized TF-IDF
    tfidf_docs, idf_dict = compute_tfidf_vectorized([])
    assert tfidf_docs == []
    assert idf_dict == {} 