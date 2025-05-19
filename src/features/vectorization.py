"""
Vectorization utilities for text data.

This module contains functions for converting text data into numerical features
using various vectorization techniques like TF-IDF.
"""

from collections import Counter, defaultdict
import math
import numpy as np
from typing import List, Dict, Tuple

def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """
    Compute Term Frequency (TF) for a list of tokens.

    This function calculates the normalized term frequency for each unique token
    in the input list. The frequency is normalized by the total number of tokens,
    giving the proportion of each term in the document.

    Parameters
    ----------
    tokens : List[str]
        A list of tokenized words from a document.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping each unique term to its normalized frequency.
    Examples
    --------
    >>> tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
    >>> compute_tf(tokens)
    {'the': 0.2222222222222222, 'quick': 0.1111111111111111, 'brown': 0.1111111111111111,
     'fox': 0.1111111111111111, 'jumps': 0.1111111111111111, 'over': 0.1111111111111111,
     'lazy': 0.1111111111111111, 'dog': 0.1111111111111111}

    Notes
    -----
    - The sum of all frequencies will equal 1.0
    - Terms that appear more frequently will have higher values
    - This is a basic TF implementation that doesn't account for document length
      or term importance
    """
    # Calculate total number of tokens
    total_count = len(tokens)
    
    # Count occurrences of each term
    terms_count = Counter(tokens)
    
    # Calculate normalized term frequencies
    tf_doc = {term: count / total_count for term, count in terms_count.items()}
    
    return tf_doc

def compute_idf(corpus: List[List[str]]) -> Dict[str, float]:
    """
    Compute Inverse Document Frequency (IDF) for a corpus of documents.

    This function calculates the IDF score for each unique term in the corpus.
    IDF measures how important a term is across the entire corpus by calculating
    the logarithm of the ratio of total documents to the number of documents
    containing the term.

    Parameters
    ----------
    corpus : List[List[str]]
        A list of documents, where each document is a list of tokenized words.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping each unique term to its IDF score.
    Examples
    --------
    >>> corpus = [
    ...     ['the', 'quick', 'brown', 'fox'],
    ...     ['the', 'lazy', 'dog'],
    ...     ['the', 'quick', 'dog']
    ... ]
    >>> compute_idf(corpus)
    {'the': 0.0, 'quick': 0.4054651081081644, 'brown': 1.0986122886681098,
     'fox': 1.0986122886681098, 'lazy': 1.0986122886681098, 'dog': 0.4054651081081644}

    Notes
    -----
    - Terms that appear in all documents will have an IDF of 0
    - Terms that appear in fewer documents will have higher IDF scores
    - The IDF score increases as the term becomes more rare across the corpus
    - This implementation uses natural logarithm (base e)
    """
    # Get total number of documents
    N = len(corpus)
    
    # Initialize document frequency counter
    df_counts = defaultdict(int)
    
    # Count document frequency for each term
    for doc in corpus:
        unique_terms = set(doc)  # Get unique terms in document
        for term in unique_terms:
            df_counts[term] += 1
    
    # Compute IDF scores
    idf_doc = {
        term: math.log(N / count) 
        for term, count in df_counts.items()
    }
    
    return idf_doc

def compute_tfidf(
    tf_dict: List[Dict[str, float]], 
    idf_dict: Dict[str, float]
) -> List[Dict[str, float]]:
    """
    Compute TF-IDF scores for a corpus of documents.

    This function calculates the TF-IDF score for each term in each document by
    multiplying the Term Frequency (TF) with the Inverse Document Frequency (IDF).

    Parameters
    ----------
    tf_dict : List[Dict[str, float]]
        A list of dictionaries containing term frequencies for each document.
    idf_dict : Dict[str, float]
        A dictionary containing IDF scores for each unique term in the corpus.

    Returns
    -------
    List[Dict[str, float]]
        A list of dictionaries containing TF-IDF scores for each document.
    Examples
    --------
    >>> tf_dict = [
    ...     {'the': 0.2, 'quick': 0.1, 'brown': 0.1, 'fox': 0.1},
    ...     {'the': 0.2, 'lazy': 0.1, 'dog': 0.1}
    ... ]
    >>> idf_dict = {
    ...     'the': 0.0, 'quick': 0.405, 'brown': 1.099,
    ...     'fox': 1.099, 'lazy': 1.099, 'dog': 0.405
    ... }
    >>> compute_tfidf(tf_dict, idf_dict)
    [
        {'the': 0.0, 'quick': 0.0405, 'brown': 0.1099, 'fox': 0.1099},
        {'the': 0.0, 'lazy': 0.1099, 'dog': 0.0405}
    ]

    Notes
    -----
    - TF-IDF score is calculated as: TF * IDF
    - Higher scores indicate terms that are more important to the document
    - Terms that appear in all documents (high IDF) will have lower scores
    - Terms that appear frequently in a document (high TF) will have higher scores
    - The function assumes all terms in tf_dict exist in idf_dict   
    """
    # Initialize list to store TF-IDF scores for each document
    tfidf_corpus = []
    
    # Calculate TF-IDF scores for each document
    for tf_doc in tf_dict:
        # Multiply TF and IDF scores for each term in the document
        tfidf_doc = {
            term: tf_doc[term] * idf_dict[term] 
            for term in tf_doc
        }
        tfidf_corpus.append(tfidf_doc)
    
    return tfidf_corpus 


def compute_tf_vectorized(tokens: List[str]) -> Dict[str, float]:
    """
    Compute Term Frequency (TF) for a document using vectorized operations.

    This function calculates the normalized term frequency for each unique token
    in the input document using NumPy's vectorized operations.

    Parameters
    ----------
    tokens : List[str]
        A list of tokenized words from a document.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping each unique term to its normalized frequency.

    Examples
    --------
    >>> tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
    >>> compute_tf_vectorized(tokens)
    {'the': 0.2222222222222222, 'quick': 0.1111111111111111, 'brown': 0.1111111111111111,
     'fox': 0.1111111111111111, 'jumps': 0.1111111111111111, 'over': 0.1111111111111111,
     'lazy': 0.1111111111111111, 'dog': 0.1111111111111111}
    """
    # Count occurrences of each term using Counter
    term_counts = Counter(tokens)
    
    # Calculate total number of tokens
    total_tokens = len(tokens)
    
    # Extract terms and counts into arrays
    terms = np.array(list(term_counts.keys()))
    counts = np.array(list(term_counts.values()), dtype=float)
    
    # Vectorized division to calculate term frequencies
    frequencies = counts / total_tokens
    
    # Create and return the term frequency dictionary
    return dict(zip(terms, frequencies))


def compute_idf_vectorized(corpus: List[List[str]]) -> Dict[str, float]:
    """
    Compute Inverse Document Frequency (IDF) for a corpus using vectorized operations.

    This function calculates the IDF score for each unique term in the corpus using
    NumPy's vectorized operations.

    Parameters
    ----------
    corpus : List[List[str]]
        A list of documents, where each document is a list of tokenized words.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping each unique term to its IDF score.

    Examples
    --------
    >>> corpus = [
    ...     ['the', 'quick', 'brown', 'fox'],
    ...     ['the', 'lazy', 'dog'],
    ...     ['the', 'quick', 'dog']
    ... ]
    >>> compute_idf_vectorized(corpus)
    {'the': 0.0, 'quick': 0.4054651081081644, 'brown': 1.0986122886681098,
     'fox': 1.0986122886681098, 'lazy': 1.0986122886681098, 'dog': 0.4054651081081644}
    """
    # Get total number of documents
    n_docs = len(corpus)
    
    # Create document frequency counter
    doc_freq = Counter()
    
    # Count document frequency for each term (only count once per document)
    for doc in corpus:
        # Add 1 for each unique term in the document
        doc_freq.update(set(doc))
    
    # Extract terms and counts into arrays
    terms = np.array(list(doc_freq.keys()))
    counts = np.array(list(doc_freq.values()))
    
    # Vectorized IDF calculation: log(N/df)
    idf_values = np.log(n_docs / counts)
    
    # Create and return the IDF dictionary
    return dict(zip(terms, idf_values))


def compute_tfidf_vectorized(corpus: List[List[str]]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Compute TF-IDF for a corpus using vectorized operations.

    This function calculates both term frequency (TF) for each document and
    inverse document frequency (IDF) for the entire corpus, then combines them
    to create TF-IDF scores.

    Parameters
    ----------
    corpus : List[List[str]]
        A list of documents, where each document is a list of tokenized words.

    Returns
    -------
    Tuple[List[Dict[str, float]], Dict[str, float]]
        A tuple containing:
        - A list of dictionaries, where each dictionary maps terms to their TF-IDF scores
          for a particular document
        - The IDF dictionary mapping terms to their corpus-wide IDF scores

    Examples
    --------
    >>> corpus = [
    ...     ['the', 'quick', 'brown', 'fox'],
    ...     ['the', 'lazy', 'dog'],
    ...     ['the', 'quick', 'dog']
    ... ]
    >>> tfidf_docs, idf_scores = compute_tfidf_vectorized(corpus)
    >>> tfidf_docs[0]  # TF-IDF scores for first document
    {'the': 0.0, 'quick': 0.1013662770270411, 'brown': 0.27465307216702745, 'fox': 0.27465307216702745}
    """
    # Compute IDF scores for the corpus
    idf_scores = compute_idf_vectorized(corpus)
    
    # Compute TF-IDF for each document
    tfidf_docs = []
    
    for doc in corpus:
        # Compute TF for this document
        tf_scores = compute_tf_vectorized(doc)
        
        # Compute TF-IDF by multiplying TF and IDF
        tfidf = {}
        for term, tf in tf_scores.items():
            # Only include terms that exist in the corpus vocabulary
            if term in idf_scores:
                tfidf[term] = tf * idf_scores[term]
        
        tfidf_docs.append(tfidf)
    
    return tfidf_docs, idf_scores

