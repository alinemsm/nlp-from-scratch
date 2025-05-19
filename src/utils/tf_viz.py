"""
TF (Term Frequency) visualization and validation utilities.

This module contains specialized functions to visualize and validate
term frequency computations in text analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Union, Optional, Callable
import math


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """
    Compute term frequency for a document.
    
    Args:
        tokens: List of tokens in a document
        
    Returns:
        Dictionary with term frequencies
    """
    tf_dict = {}
    token_counts = Counter(tokens)
    doc_length = len(tokens)
    
    for token, count in token_counts.items():
        tf_dict[token] = count / doc_length
        
    return tf_dict


def compute_tf_for_corpus(corpus: List[List[str]]) -> List[Dict[str, float]]:
    """
    Compute term frequencies for all documents in a corpus.
    
    Args:
        corpus: List of tokenized documents
        
    Returns:
        List of dictionaries with term frequencies for each document
    """
    return [compute_tf(doc) for doc in corpus]


def validate_tf_calculation(tokens: List[str], tf_dict: Dict[str, float]) -> Dict:
    """
    Validate that TF values are correctly calculated.
    
    Args:
        tokens: List of tokens in the document
        tf_dict: TF dictionary to validate
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        "is_valid": True,
        "errors": [],
        "token_counts": {},
        "expected_tf": {},
        "actual_tf": {}
    }
    
    # Calculate expected TF values
    token_counts = Counter(tokens)
    doc_length = len(tokens)
    expected_tf = {token: count/doc_length for token, count in token_counts.items()}
    
    # Check each token in the expected TF
    for token, expected_value in expected_tf.items():
        # Check if token exists in provided TF dict
        if token not in tf_dict:
            validation["is_valid"] = False
            validation["errors"].append(f"Missing token: {token}")
            validation["token_counts"][token] = token_counts[token]
            validation["expected_tf"][token] = expected_value
            validation["actual_tf"][token] = "Missing"
            continue
            
        # Check if TF value is correct
        actual_value = tf_dict[token]
        if not math.isclose(actual_value, expected_value, rel_tol=1e-10):
            validation["is_valid"] = False
            validation["errors"].append(f"Incorrect TF for token: {token}")
            validation["token_counts"][token] = token_counts[token]
            validation["expected_tf"][token] = expected_value
            validation["actual_tf"][token] = actual_value
    
    # Check for extra tokens in provided TF dict
    for token in tf_dict:
        if token not in expected_tf:
            validation["is_valid"] = False
            validation["errors"].append(f"Extra token: {token}")
            validation["token_counts"][token] = token_counts.get(token, 0)
            validation["expected_tf"][token] = "Should not exist"
            validation["actual_tf"][token] = tf_dict[token]
    
    return validation


def plot_tf_values(tf_dict: Dict[str, float], top_n: int = 20, 
                  sort: bool = True, figsize: Tuple[int, int] = (12, 6),
                  title: str = "Term Frequencies", plot_type: str = 'bar') -> None:
    """
    Plot term frequency values for a document.
    
    Args:
        tf_dict: Dictionary with term frequencies
        top_n: Number of top terms to display
        sort: Whether to sort terms by frequency
        figsize: Figure size (width, height)
        title: Plot title
        plot_type: 'bar' for vertical bars or 'horizontal' for horizontal bars
    """
    # Sort terms if requested and take top N
    if sort:
        tf_items = sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    else:
        tf_items = list(tf_dict.items())[:top_n]
    
    tokens = [item[0] for item in tf_items]
    frequencies = [item[1] for item in tf_items]
    
    plt.figure(figsize=figsize)
    
    if plot_type == 'horizontal':
        # Horizontal bar chart (better for longer token names)
        bars = plt.barh(range(len(tokens)), frequencies, alpha=0.7)
        plt.yticks(range(len(tokens)), tokens)
        plt.xlabel('Term Frequency')
        
        # Add value labels
        for bar, freq in zip(bars, frequencies):
            plt.text(freq + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{freq:.4f}', va='center')
            
    else:
        # Vertical bar chart
        bars = plt.bar(range(len(tokens)), frequencies, alpha=0.7)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.ylabel('Term Frequency')
        
        # Add value labels
        for bar, freq in zip(bars, frequencies):
            plt.text(bar.get_x() + bar.get_width()/2, freq + 0.005, 
                   f'{freq:.4f}', ha='center', va='bottom', rotation=90 if len(tokens) > 10 else 0)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_tf_comparison(tokens: List[str], tf_functions: Dict[str, Callable], 
                       top_n: int = 10, figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    Compare different TF calculation methods for the same document.
    
    Args:
        tokens: List of tokens in a document
        tf_functions: Dictionary mapping method names to TF calculation functions
        top_n: Number of top terms to display
        figsize: Figure size (width, height)
    """
    # Calculate TF using each method
    results = {}
    for method_name, tf_func in tf_functions.items():
        results[method_name] = tf_func(tokens)
    
    # Get the union of all tokens from all methods
    all_tokens = set()
    for tf_dict in results.values():
        all_tokens.update(tf_dict.keys())
    
    # Create a DataFrame for comparison
    df = pd.DataFrame(index=all_tokens)
    for method_name, tf_dict in results.items():
        df[method_name] = df.index.map(lambda x: tf_dict.get(x, 0))
    
    # Sort by the first method's values
    first_method = list(tf_functions.keys())[0]
    df = df.sort_values(by=first_method, ascending=False).head(top_n)
    
    # Plot comparison
    plt.figure(figsize=figsize)
    
    # Set positions on x-axis
    x = np.arange(len(df.index))
    width = 0.8 / len(results)  # Width of bars with space between groups
    
    # Plot bars for each method
    for i, method_name in enumerate(results.keys()):
        offset = (i - len(results)/2 + 0.5) * width
        bars = plt.bar(x + offset, df[method_name], width, alpha=0.7, label=method_name)
    
    # Labels and title
    plt.xlabel('Tokens')
    plt.ylabel('Term Frequency')
    plt.title('TF Calculation Comparison')
    plt.xticks(x, df.index, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_tf_heatmap(corpus_tf: List[Dict[str, float]], top_n: int = 20, 
                   figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Create a heatmap of term frequencies across documents.
    
    Args:
        corpus_tf: List of TF dictionaries, one per document
        top_n: Number of top terms to display
        figsize: Figure size (width, height)
    """
    # Get all unique tokens
    all_tokens = set()
    for tf_dict in corpus_tf:
        all_tokens.update(tf_dict.keys())
    
    # Create a DataFrame for the heatmap
    df = pd.DataFrame(index=list(all_tokens))
    
    # Add each document's TF values
    for i, tf_dict in enumerate(corpus_tf):
        df[f'Doc {i}'] = df.index.map(lambda x: tf_dict.get(x, 0))
    
    # Find top tokens across documents
    if len(all_tokens) > top_n:
        # Sort by row sum (total TF across all documents)
        df['total'] = df.sum(axis=1)
        df = df.sort_values(by='total', ascending=False).head(top_n)
        df = df.drop(columns=['total'])
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Term Frequency Heatmap Across Documents')
    plt.tight_layout()
    plt.show()


def plot_tf_distribution(tf_dict: Dict[str, float], bins: int = 20,
                        figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the distribution of term frequency values.
    
    Args:
        tf_dict: Dictionary with term frequencies
        bins: Number of histogram bins
        figsize: Figure size (width, height)
    """
    values = list(tf_dict.values())
    
    plt.figure(figsize=figsize)
    plt.hist(values, bins=bins, alpha=0.7)
    plt.xlabel('Term Frequency Value')
    plt.ylabel('Number of Terms')
    plt.title('Distribution of Term Frequency Values')
    
    # Add statistics to the plot
    mean_val = np.mean(values)
    median_val = np.median(values)
    max_val = max(values)
    min_val = min(values)
    
    stats_text = (f"Mean: {mean_val:.4f}\n"
                 f"Median: {median_val:.4f}\n"
                 f"Max: {max_val:.4f}\n"
                 f"Min: {min_val:.4f}")
    
    # Position the text in the upper right
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def validate_tf_sum_to_one(corpus_tf: List[Dict[str, float]], 
                           tol: float = 1e-10) -> List[Dict]:
    """
    Validate that TF values sum to 1 for each document.
    
    Args:
        corpus_tf: List of TF dictionaries for each document
        tol: Tolerance for floating point comparison
        
    Returns:
        List of validation results for each document
    """
    results = []
    
    for i, tf_dict in enumerate(corpus_tf):
        sum_tf = sum(tf_dict.values())
        is_valid = math.isclose(sum_tf, 1.0, rel_tol=tol)
        
        results.append({
            "doc_index": i,
            "sum_tf": sum_tf,
            "is_valid": is_valid,
            "error": abs(sum_tf - 1.0)
        })
    
    return results


def plot_tf_sum_validation(corpus_tf: List[Dict[str, float]], 
                          figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot validation that TF values sum to 1 for each document.
    
    Args:
        corpus_tf: List of TF dictionaries for each document
        figsize: Figure size (width, height)
    """
    # Validate TF sums
    validations = validate_tf_sum_to_one(corpus_tf)
    
    # Extract values for plotting
    doc_indices = [v["doc_index"] for v in validations]
    sums = [v["sum_tf"] for v in validations]
    errors = [v["error"] for v in validations]
    valid_status = [v["is_valid"] for v in validations]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot the actual sums
    bars = ax1.bar(doc_indices, sums, alpha=0.7)
    
    # Color bars based on validity
    for i, bar in enumerate(bars):
        bar.set_color('green' if valid_status[i] else 'red')
    
    # Add horizontal line at y=1
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(sums):
        ax1.text(i, v + 0.01, f'{v:.6f}', ha='center', va='bottom', 
                rotation=90 if len(corpus_tf) > 10 else 0)
    
    ax1.set_title('Sum of TF Values for Each Document')
    ax1.set_ylabel('Sum of TF Values')
    ax1.set_xticks(doc_indices)
    ax1.set_xticklabels([f'Doc {i}' for i in doc_indices])
    
    # Plot the errors
    ax2.bar(doc_indices, errors, alpha=0.7, color='purple')
    ax2.set_yscale('log')  # Log scale for better visualization of small errors
    ax2.set_title('Error (|Sum - 1.0|)')
    ax2.set_ylabel('Error (log scale)')
    ax2.set_xticks(doc_indices)
    ax2.set_xticklabels([f'Doc {i}' for i in doc_indices])
    
    # Add value labels for errors
    for i, v in enumerate(errors):
        if v > 0:  # Only label non-zero errors
            ax2.text(i, v * 1.1, f'{v:.2e}', ha='center', va='bottom', 
                    rotation=90 if len(corpus_tf) > 10 else 0)
    
    # Add summary text
    num_valid = sum(valid_status)
    summary_text = (
        f"Documents with TF sum = 1.0: {num_valid}/{len(corpus_tf)}\n"
        f"Average error: {np.mean(errors):.2e}\n"
        f"Max error: {max(errors):.2e}"
    )
    
    # Position the text in the upper right of the top plot
    ax1.text(0.95, 0.95, summary_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def plot_document_tf_similarity(corpus_tf: List[Dict[str, float]], 
                               method: str = 'cosine',
                               figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Create a heatmap showing similarity between documents based on TF values.
    
    Args:
        corpus_tf: List of TF dictionaries for the corpus
        method: Similarity method ('cosine', 'euclidean', or 'jaccard')
        figsize: Figure size (width, height)
    """
    num_docs = len(corpus_tf)
    similarity_matrix = np.zeros((num_docs, num_docs))
    
    # Pre-process: convert TF dicts to fixed-dimension vectors
    all_tokens = set()
    for tf_dict in corpus_tf:
        all_tokens.update(tf_dict.keys())
    
    # Create vectors for each document
    vectors = []
    for tf_dict in corpus_tf:
        vector = np.array([tf_dict.get(token, 0) for token in all_tokens])
        vectors.append(vector)
    
    # Compute pairwise similarity
    for i in range(num_docs):
        for j in range(num_docs):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity
            else:
                if method == 'cosine':
                    # Cosine similarity
                    dot_product = np.dot(vectors[i], vectors[j])
                    norm_i = np.linalg.norm(vectors[i])
                    norm_j = np.linalg.norm(vectors[j])
                    if norm_i > 0 and norm_j > 0:
                        similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
                    else:
                        similarity_matrix[i, j] = 0
                        
                elif method == 'euclidean':
                    # Euclidean distance (converted to similarity)
                    distance = np.linalg.norm(vectors[i] - vectors[j])
                    similarity_matrix[i, j] = 1 / (1 + distance)  # Convert to similarity
                    
                elif method == 'jaccard':
                    # Jaccard similarity (on token presence, not TF values)
                    tokens_i = set(corpus_tf[i].keys())
                    tokens_j = set(corpus_tf[j].keys())
                    intersection = len(tokens_i & tokens_j)
                    union = len(tokens_i | tokens_j)
                    similarity_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Plot the similarity matrix
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='viridis',
               xticklabels=[f'Doc {i}' for i in range(num_docs)],
               yticklabels=[f'Doc {i}' for i in range(num_docs)])
    plt.title(f'Document Similarity Based on TF ({method.capitalize()} Similarity)')
    plt.tight_layout()
    plt.show()


def plot_tf_idf_comparison(tf_dict: Dict[str, float], tf_idf_dict: Dict[str, float],
                          top_n: int = 15, figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Compare TF and TF-IDF values for the same document.
    
    Args:
        tf_dict: Dictionary with term frequencies
        tf_idf_dict: Dictionary with TF-IDF values
        top_n: Number of top terms to display
        figsize: Figure size (width, height)
    """
    # Get union of tokens
    all_tokens = set(tf_dict.keys()) | set(tf_idf_dict.keys())
    
    # Create a DataFrame for comparison
    df = pd.DataFrame(index=list(all_tokens))
    df['TF'] = df.index.map(lambda x: tf_dict.get(x, 0))
    df['TF-IDF'] = df.index.map(lambda x: tf_idf_dict.get(x, 0))
    
    # Sort by TF-IDF values and take top N
    df = df.sort_values(by='TF-IDF', ascending=False).head(top_n)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot TF values
    bars1 = ax1.bar(range(len(df.index)), df['TF'], alpha=0.7)
    ax1.set_ylabel('Term Frequency')
    ax1.set_title('Term Frequency (TF)')
    
    # Add value labels to TF plot
    for bar, value in zip(bars1, df['TF']):
        ax1.text(bar.get_x() + bar.get_width()/2, value + 0.005, 
                f'{value:.4f}', ha='center', va='bottom')
    
    # Plot TF-IDF values
    bars2 = ax2.bar(range(len(df.index)), df['TF-IDF'], alpha=0.7)
    ax2.set_ylabel('TF-IDF Value')
    ax2.set_title('TF-IDF')
    
    # Add value labels to TF-IDF plot
    for bar, value in zip(bars2, df['TF-IDF']):
        ax2.text(bar.get_x() + bar.get_width()/2, value + 0.005, 
                f'{value:.4f}', ha='center', va='bottom')
    
    # Set x-ticks for both plots
    for ax in [ax1, ax2]:
        ax.set_xticks(range(len(df.index)))
        ax.set_xticklabels(df.index, rotation=45, ha='right')
    
    plt.suptitle('Comparison of TF and TF-IDF Values', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the overall title
    plt.show() 