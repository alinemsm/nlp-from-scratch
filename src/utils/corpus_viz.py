"""
Corpus visualization and validation utilities.

This module contains functions to help visualize and validate
processed text corpora (tokenized documents).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional, Union
from wordcloud import WordCloud  # You may need to install this: pip install wordcloud


def basic_corpus_stats(processed_corpus: List[List[str]]) -> Dict:
    """
    Get basic statistics about a processed corpus.
    
    Args:
        processed_corpus: List of tokenized documents
        
    Returns:
        Dictionary with basic corpus statistics
    """
    doc_lengths = [len(doc) for doc in processed_corpus]
    
    # Flatten corpus to get all tokens
    all_tokens = [token for doc in processed_corpus for token in doc]
    token_counts = Counter(all_tokens)
    
    return {
        "num_documents": len(processed_corpus),
        "vocab_size": len(token_counts),
        "total_tokens": len(all_tokens),
        "unique_tokens": len(token_counts),
        "doc_length_min": min(doc_lengths),
        "doc_length_max": max(doc_lengths),
        "doc_length_avg": sum(doc_lengths) / len(doc_lengths),
        "token_counts": token_counts,
        "doc_lengths": doc_lengths
    }


def plot_token_distribution(token_counts: Counter, top_n: int = 20, 
                            plot_zipf: bool = True, figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot token frequency distribution.
    
    Args:
        token_counts: Counter object with token frequencies
        top_n: Number of top tokens to display
        plot_zipf: Whether to plot Zipf's law distribution
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # If plot_zipf is True, create a subplot layout
    if plot_zipf:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot top tokens
        tokens = [token for token, _ in token_counts.most_common(top_n)]
        freqs = [count for _, count in token_counts.most_common(top_n)]
        
        ax1.bar(range(len(tokens)), freqs)
        ax1.set_xticks(range(len(tokens)))
        ax1.set_xticklabels(tokens, rotation=45, ha='right')
        ax1.set_title(f'Top {top_n} Most Frequent Tokens')
        ax1.set_ylabel('Frequency')
        
        # Plot Zipf's law distribution
        token_freq = pd.Series(token_counts).sort_values(ascending=False)
        
        ax2.bar(range(len(token_freq)), token_freq.values)
        ax2.set_yscale('log')
        ax2.set_xlabel('Token Rank')
        ax2.set_ylabel('Frequency (log scale)')
        ax2.set_title('Token Frequency Distribution (Zipf\'s Law)')
        
        plt.tight_layout()
        plt.show()
    else:
        # Just plot top tokens
        tokens = [token for token, _ in token_counts.most_common(top_n)]
        freqs = [count for _, count in token_counts.most_common(top_n)]
        
        plt.bar(range(len(tokens)), freqs)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.title(f'Top {top_n} Most Frequent Tokens')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()


def plot_doc_lengths(doc_lengths: List[int], figsize: Tuple[int, int] = (10, 6), 
                    plot_type: str = 'bar', sort: bool = False) -> None:
    """
    Plot document lengths with document identification.
    
    Args:
        doc_lengths: List of document lengths
        figsize: Figure size (width, height)
        plot_type: 'bar' for individual document bars or 'hist' for histogram
        sort: Whether to sort documents by length
    """
    plt.figure(figsize=figsize)
    
    if plot_type == 'hist':
        # Traditional histogram
        plt.hist(doc_lengths, bins=20, alpha=0.7)
        plt.axvline(sum(doc_lengths)/len(doc_lengths), color='red', 
                    linestyle='dashed', linewidth=1)
        plt.xlabel('Document Length (tokens)')
        plt.ylabel('Number of Documents')
        plt.title('Distribution of Document Lengths')
    else:
        # Bar chart with document identification
        doc_indices = list(range(len(doc_lengths)))
        
        # Optionally sort by length
        if sort:
            indices_lengths = sorted(zip(doc_indices, doc_lengths), key=lambda x: x[1], reverse=True)
            doc_indices = [idx for idx, _ in indices_lengths]
            doc_lengths = [length for _, length in indices_lengths]
        
        # Create bar chart
        bars = plt.bar(doc_indices, doc_lengths, alpha=0.7)
        
        # Add value labels on top of bars
        for i, (bar, length) in enumerate(zip(bars, doc_lengths)):
            plt.text(bar.get_x() + bar.get_width()/2, length + 0.5, 
                    f'Doc {doc_indices[i]}: {length}', 
                    ha='center', va='bottom', rotation=90 if len(doc_lengths) > 10 else 0)
        
        # Add average line
        avg_length = sum(doc_lengths)/len(doc_lengths)
        plt.axhline(avg_length, color='red', linestyle='dashed', linewidth=1)
        plt.text(len(doc_lengths)-1, avg_length, f'Avg: {avg_length:.1f}', 
                 ha='right', va='bottom', color='red')
        
        # Labels
        plt.xlabel('Document Index')
        plt.ylabel('Length (tokens)')
        plt.title('Document Lengths')
        
        # Adjust x-ticks for better readability
        if len(doc_lengths) <= 20:
            plt.xticks(doc_indices, [f'Doc {i}' for i in doc_indices])
        else:
            # Show fewer tick labels for readability with many documents
            step = max(1, len(doc_lengths) // 10)
            plt.xticks([i for i in doc_indices if i % step == 0], 
                      [f'Doc {i}' for i in doc_indices if i % step == 0])
    
    plt.tight_layout()
    plt.show()


def plot_term_frequencies(processed_corpus: List[List[str]], doc_indices: Union[int, List[int]] = None, 
                          top_n: int = 20, sort: bool = True, 
                          plot_type: str = 'bar', figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize term frequencies for one or more documents with clear document identification.
    
    Args:
        processed_corpus: List of tokenized documents
        doc_indices: Document index or list of indices to visualize (None = all documents)
        top_n: Number of top terms to display per document
        sort: Whether to sort terms by frequency (highest first)
        plot_type: 'bar' for bar chart or 'horizontal' for horizontal bars
        figsize: Figure size (width, height)
    """
    # Handle single document case
    if isinstance(doc_indices, int):
        doc_indices = [doc_indices]
    
    # If None, use all documents (up to a reasonable limit)
    if doc_indices is None:
        if len(processed_corpus) > 6:
            print(f"Too many documents ({len(processed_corpus)}) to display all. Showing first 6.")
            doc_indices = list(range(6))
        else:
            doc_indices = list(range(len(processed_corpus)))
    
    num_docs = len(doc_indices)
    
    # Calculate number of rows and columns for subplot grid
    if num_docs <= 3:
        n_rows, n_cols = 1, num_docs
    else:
        n_rows = (num_docs + 1) // 2  # Ceiling division
        n_cols = min(2, num_docs)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Process each document
    for i, doc_idx in enumerate(doc_indices):
        if doc_idx >= len(processed_corpus):
            print(f"Warning: Document index {doc_idx} is out of range. Skipping.")
            continue
            
        # Get document and compute term frequencies
        doc = processed_corpus[doc_idx]
        tf_dict = {}
        doc_counter = Counter(doc)
        doc_length = len(doc)
        
        # Calculate term frequency for each token
        for token, count in doc_counter.items():
            tf_dict[token] = count / doc_length
        
        # Sort terms if requested
        if sort:
            tf_items = sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        else:
            tf_items = list(tf_dict.items())[:top_n]
        
        tokens = [item[0] for item in tf_items]
        frequencies = [item[1] for item in tf_items]
        
        # Create the plot
        ax = axes[i]
        
        if plot_type == 'horizontal':
            # Horizontal bar chart (better for longer token names)
            bars = ax.barh(range(len(tokens)), frequencies, alpha=0.7)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens)
            ax.set_xlabel('Term Frequency')
            
            # Add value labels
            for bar, freq in zip(bars, frequencies):
                ax.text(freq + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{freq:.3f}', va='center')
                
        else:
            # Vertical bar chart
            bars = ax.bar(range(len(tokens)), frequencies, alpha=0.7)
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_ylabel('Term Frequency')
            
            # Add value labels
            for bar, freq in zip(bars, frequencies):
                ax.text(bar.get_x() + bar.get_width()/2, freq + 0.01, 
                       f'{freq:.3f}', ha='center', va='bottom', rotation=90 if len(tokens) > 10 else 0)
        
        # Set title with document info
        ax.set_title(f'Document {doc_idx} (length: {doc_length})')
        
    # Hide unused subplots if any
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Term Frequencies by Document', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the overall title
    plt.show()


def create_wordcloud(token_counts: Counter, figsize: Tuple[int, int] = (12, 6),
                     max_words: int = 100, width: int = 800, height: int = 400) -> None:
    """
    Create and display a word cloud from token counts.
    
    Args:
        token_counts: Counter object with token frequencies
        figsize: Figure size (width, height)
        max_words: Maximum number of words to include
        width: Width of the word cloud image
        height: Height of the word cloud image
    """
    wordcloud = WordCloud(width=width, height=height, background_color='white', 
                          max_words=max_words, contour_width=3, contour_color='steelblue')
    wordcloud.generate_from_frequencies(token_counts)
    
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Word Cloud')
    plt.tight_layout()
    plt.show()


def validate_preprocessing(token_counts: Counter, check_stopwords: bool = True,
                          check_case: bool = True, check_punct: bool = True) -> Dict:
    """
    Validate preprocessing steps by checking for stopwords, case, and punctuation.
    
    Args:
        token_counts: Counter object with token frequencies
        check_stopwords: Whether to check for common stopwords
        check_case: Whether to check for uppercase characters
        check_punct: Whether to check for punctuation
        
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Check for stopwords
    if check_stopwords:
        common_stopwords = ['the', 'a', 'an', 'of', 'and', 'in', 'to', 'is', 
                            'that', 'it', 'with', 'as', 'for']
        stopwords_present = [word for word in common_stopwords if word in token_counts]
        stopwords_counts = {word: token_counts[word] for word in stopwords_present}
        results['stopwords'] = {
            'present': len(stopwords_present) > 0,
            'words': stopwords_counts
        }
    
    # Check for uppercase characters
    if check_case:
        uppercase_tokens = [token for token in token_counts if any(c.isupper() for c in token)]
        uppercase_counts = {token: token_counts[token] for token in uppercase_tokens[:10]}
        results['uppercase'] = {
            'present': len(uppercase_tokens) > 0,
            'count': len(uppercase_tokens),
            'examples': uppercase_counts
        }
    
    # Check for punctuation
    if check_punct:
        punct_chars = set('.,;:!?()[]{}"\'-')
        punct_tokens = [token for token in token_counts if any(c in punct_chars for c in token)]
        punct_counts = {token: token_counts[token] for token in punct_tokens[:10]}
        results['punctuation'] = {
            'present': len(punct_tokens) > 0,
            'count': len(punct_tokens),
            'examples': punct_counts
        }
    
    return results


def compute_similarity_matrix(processed_corpus: List[List[str]], n_docs: Optional[int] = None) -> np.ndarray:
    """
    Compute document similarity matrix using Jaccard similarity.
    
    Args:
        processed_corpus: List of tokenized documents
        n_docs: Number of documents to include (defaults to all)
        
    Returns:
        Numpy array with similarity matrix
    """
    # Limit to a subset of documents if needed
    if n_docs is None:
        n_docs = len(processed_corpus)
    docs = processed_corpus[:n_docs]
    
    # Create sets of tokens for each document
    doc_sets = [set(doc) for doc in docs]
    
    # Compute Jaccard similarity between documents
    similarity_matrix = np.zeros((n_docs, n_docs))
    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Jaccard similarity: |A ∩ B| / |A ∪ B|
                intersection = len(doc_sets[i] & doc_sets[j])
                union = len(doc_sets[i] | doc_sets[j])
                similarity_matrix[i, j] = intersection / union if union > 0 else 0
    
    return similarity_matrix


def plot_similarity_matrix(similarity_matrix: np.ndarray, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot document similarity matrix as a heatmap.
    
    Args:
        similarity_matrix: Numpy array with similarity values
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Document Similarity Matrix (Jaccard Similarity)')
    plt.xlabel('Document Index')
    plt.ylabel('Document Index')
    plt.tight_layout()
    plt.show()


def find_unique_terms(processed_corpus: List[List[str]], n_docs: Optional[int] = None) -> List[List[str]]:
    """
    Find terms that are unique to each document.
    
    Args:
        processed_corpus: List of tokenized documents
        n_docs: Number of documents to include (defaults to all)
        
    Returns:
        List of lists containing unique terms for each document
    """
    # Limit to a subset of documents if needed
    if n_docs is None:
        n_docs = len(processed_corpus)
    docs = processed_corpus[:n_docs]
    
    # Create token sets for each document
    doc_token_sets = [set(doc) for doc in docs]
    
    # For each document, find tokens that don't appear in other documents
    unique_terms = []
    for i, doc_set in enumerate(doc_token_sets):
        other_docs = set().union(*[doc_token_sets[j] for j in range(n_docs) if j != i])
        unique = doc_set - other_docs
        unique_terms.append(list(unique))
    
    return unique_terms


def create_term_doc_matrix(processed_corpus: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
    """
    Create a term-document matrix from the processed corpus.
    
    Args:
        processed_corpus: List of tokenized documents
        
    Returns:
        Tuple of (term-document matrix, vocabulary list)
    """
    # Build vocabulary (all unique tokens)
    all_tokens = [token for doc in processed_corpus for token in doc]
    vocabulary = sorted(set(all_tokens))
    vocab_to_idx = {word: i for i, word in enumerate(vocabulary)}
    
    # Create term-document matrix
    term_doc_matrix = np.zeros((len(vocabulary), len(processed_corpus)))
    
    for doc_idx, doc in enumerate(processed_corpus):
        doc_counts = Counter(doc)
        for word, count in doc_counts.items():
            term_doc_matrix[vocab_to_idx[word], doc_idx] = count
    
    return term_doc_matrix, vocabulary


def plot_term_doc_matrix(term_doc_matrix: np.ndarray, vocabulary: List[str], 
                         n_terms: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot a term-document matrix heatmap.
    
    Args:
        term_doc_matrix: Term-document matrix
        vocabulary: List of terms
        n_terms: Number of top terms to include
        figsize: Figure size (width, height)
    """
    # If we have too many terms, just show the most frequent ones
    if len(vocabulary) > n_terms:
        # Sum across documents to get total frequency
        term_freq = term_doc_matrix.sum(axis=1)
        top_term_indices = term_freq.argsort()[-n_terms:][::-1]
        matrix_subset = term_doc_matrix[top_term_indices, :]
        vocab_subset = [vocabulary[i] for i in top_term_indices]
    else:
        matrix_subset = term_doc_matrix
        vocab_subset = vocabulary
    
    plt.figure(figsize=figsize)
    sns.heatmap(matrix_subset, cmap='viridis',
                yticklabels=vocab_subset,
                xticklabels=[f'Doc {i}' for i in range(term_doc_matrix.shape[1])])
    plt.title(f'Term-Document Matrix ({"Top " + str(n_terms) + " terms" if n_terms < len(vocabulary) else "All terms"})')
    plt.tight_layout()
    plt.show() 