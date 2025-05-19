"""
IDF (Inverse Document Frequency) visualization and validation utilities.

This module contains specialized functions to visualize and validate
IDF computations in text analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Union, Optional, Callable
import math


def compute_idf(corpus: List[List[str]], smooth: bool = True) -> Dict[str, float]:
    """
    Compute IDF for a corpus.
    
    Args:
        corpus: List of tokenized documents
        smooth: Whether to add 1 to document frequencies (prevents division by zero)
        
    Returns:
        Dictionary with IDF values
    """
    num_docs = len(corpus)
    idf_dict = {}
    
    # Count document frequency for each term
    doc_freq = defaultdict(int)
    for doc in corpus:
        # Count each term only once per document
        for term in set(doc):
            doc_freq[term] += 1
    
    # Calculate IDF
    for term, freq in doc_freq.items():
        if smooth:
            idf_dict[term] = math.log10((num_docs + 1) / (freq + 1)) + 1
        else:
            idf_dict[term] = math.log10(num_docs / freq)
    
    return idf_dict


def validate_idf_calculation(corpus: List[List[str]], idf_dict: Dict[str, float], 
                             smooth: bool = True, tol: float = 1e-10) -> Dict:
    """
    Validate that IDF values are correctly calculated.
    
    Args:
        corpus: List of tokenized documents
        idf_dict: IDF dictionary to validate
        smooth: Whether smoothing was used in the computation
        tol: Tolerance for floating point comparison
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        "is_valid": True,
        "errors": [],
        "doc_freq": {},
        "expected_idf": {},
        "actual_idf": {}
    }
    
    # Get unique terms across corpus
    all_terms = set()
    for doc in corpus:
        all_terms.update(doc)
    
    # Count document frequency for each term
    doc_freq = defaultdict(int)
    for doc in corpus:
        # Count each term only once per document
        for term in set(doc):
            doc_freq[term] += 1
    
    # Calculate expected IDF values
    num_docs = len(corpus)
    expected_idf = {}
    for term in all_terms:
        if smooth:
            expected_idf[term] = math.log10((num_docs + 1) / (doc_freq[term] + 1)) + 1
        else:
            expected_idf[term] = math.log10(num_docs / doc_freq[term])
    
    # Check each term in the expected IDF
    for term, expected_value in expected_idf.items():
        # Check if term exists in provided IDF dict
        if term not in idf_dict:
            validation["is_valid"] = False
            validation["errors"].append(f"Missing term: {term}")
            validation["doc_freq"][term] = doc_freq[term]
            validation["expected_idf"][term] = expected_value
            validation["actual_idf"][term] = "Missing"
            continue
            
        # Check if IDF value is correct
        actual_value = idf_dict[term]
        if not math.isclose(actual_value, expected_value, rel_tol=tol):
            validation["is_valid"] = False
            validation["errors"].append(f"Incorrect IDF for term: {term}")
            validation["doc_freq"][term] = doc_freq[term]
            validation["expected_idf"][term] = expected_value
            validation["actual_idf"][term] = actual_value
    
    # Check for extra terms in provided IDF dict
    for term in idf_dict:
        if term not in expected_idf:
            validation["is_valid"] = False
            validation["errors"].append(f"Extra term: {term}")
            validation["doc_freq"][term] = doc_freq.get(term, 0)
            validation["expected_idf"][term] = "Should not exist"
            validation["actual_idf"][term] = idf_dict[term]
    
    return validation


def plot_idf_values(idf_dict: Dict[str, float], top_n: int = 20, 
                   sort_ascending: bool = True, figsize: Tuple[int, int] = (12, 6),
                   title: str = "IDF Values", plot_type: str = 'bar') -> None:
    """
    Plot IDF values. By default, shows terms with lowest IDF (most common).
    
    Args:
        idf_dict: Dictionary with IDF values
        top_n: Number of terms to display
        sort_ascending: Sort by ascending IDF (True=most common terms, False=rarest terms)
        figsize: Figure size (width, height)
        title: Plot title
        plot_type: 'bar' for vertical bars or 'horizontal' for horizontal bars
    """
    # Sort terms and take top N
    if sort_ascending:
        # Lowest IDF = most common terms
        idf_items = sorted(idf_dict.items(), key=lambda x: x[1])[:top_n]
    else:
        # Highest IDF = rarest terms
        idf_items = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    terms = [item[0] for item in idf_items]
    idf_values = [item[1] for item in idf_items]
    
    plt.figure(figsize=figsize)
    
    if plot_type == 'horizontal':
        # Horizontal bar chart (better for longer term names)
        bars = plt.barh(range(len(terms)), idf_values, alpha=0.7)
        plt.yticks(range(len(terms)), terms)
        plt.xlabel('IDF Value')
        
        # Add value labels
        for bar, val in zip(bars, idf_values):
            plt.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', va='center')
            
    else:
        # Vertical bar chart
        bars = plt.bar(range(len(terms)), idf_values, alpha=0.7)
        plt.xticks(range(len(terms)), terms, rotation=45, ha='right')
        plt.ylabel('IDF Value')
        
        # Add value labels
        for bar, val in zip(bars, idf_values):
            plt.text(bar.get_x() + bar.get_width()/2, val + 0.05, 
                   f'{val:.4f}', ha='center', va='bottom', rotation=90 if len(terms) > 10 else 0)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_term_document_frequency(corpus: List[List[str]], top_n: int = 20,
                                sort_ascending: bool = True, figsize: Tuple[int, int] = (12, 6),
                                include_pct: bool = True) -> None:
    """
    Plot document frequency (number of documents containing each term).
    
    Args:
        corpus: List of tokenized documents
        top_n: Number of terms to display
        sort_ascending: Sort by ascending frequency (True=uncommon terms, False=common terms)
        figsize: Figure size (width, height)
        include_pct: Whether to show percentage of documents as well
    """
    # Count document frequency for each term
    doc_freq = defaultdict(int)
    for doc in corpus:
        # Count each term only once per document
        for term in set(doc):
            doc_freq[term] += 1
    
    # Sort terms and take top/bottom N
    if sort_ascending:
        # Lowest doc freq = most unique terms
        df_items = sorted(doc_freq.items(), key=lambda x: x[1])[:top_n]
    else:
        # Highest doc freq = most common terms
        df_items = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    terms = [item[0] for item in df_items]
    frequencies = [item[1] for item in df_items]
    
    plt.figure(figsize=figsize)
    
    # Create bar chart
    bars = plt.bar(range(len(terms)), frequencies, alpha=0.7)
    plt.xticks(range(len(terms)), terms, rotation=45, ha='right')
    plt.ylabel('Document Frequency')
    
    # Add value labels
    num_docs = len(corpus)
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        label_text = f'{freq}'
        if include_pct:
            pct = freq / num_docs * 100
            label_text += f' ({pct:.1f}%)'
            
        plt.text(bar.get_x() + bar.get_width()/2, freq + 0.5, 
               label_text, ha='center', va='bottom', rotation=90 if len(terms) > 10 else 0)
    
    # Set title based on sort order
    if sort_ascending:
        plt.title(f'Least Common Terms in Corpus (Lowest Document Frequency)')
    else:
        plt.title(f'Most Common Terms in Corpus (Highest Document Frequency)')
    
    plt.tight_layout()
    plt.show()


def plot_idf_distribution(idf_dict: Dict[str, float], bins: int = 20,
                         figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the distribution of IDF values.
    
    Args:
        idf_dict: Dictionary with IDF values
        bins: Number of histogram bins
        figsize: Figure size (width, height)
    """
    values = list(idf_dict.values())
    
    plt.figure(figsize=figsize)
    plt.hist(values, bins=bins, alpha=0.7)
    plt.xlabel('IDF Value')
    plt.ylabel('Number of Terms')
    plt.title('Distribution of IDF Values')
    
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


def plot_idf_comparison(corpus: List[List[str]], idf_functions: Dict[str, Callable], 
                      top_n: int = 10, figsize: Tuple[int, int] = (14, 8),
                      sort_by: str = None) -> None:
    """
    Compare different IDF calculation methods.
    
    Args:
        corpus: List of tokenized documents
        idf_functions: Dictionary mapping method names to IDF calculation functions
        top_n: Number of top terms to display
        figsize: Figure size (width, height)
        sort_by: Which method to sort by (default: first method)
    """
    # Calculate IDF using each method
    results = {}
    for method_name, idf_func in idf_functions.items():
        results[method_name] = idf_func(corpus)
    
    # Get the union of all terms from all methods
    all_terms = set()
    for idf_dict in results.values():
        all_terms.update(idf_dict.keys())
    
    # Create a DataFrame for comparison
    df = pd.DataFrame(index=all_terms)
    for method_name, idf_dict in results.items():
        df[method_name] = df.index.map(lambda x: idf_dict.get(x, 0))
    
    # Sort by the specified method's values (or the first method if not specified)
    sort_method = sort_by if sort_by in results else list(idf_functions.keys())[0]
    df = df.sort_values(by=sort_method, ascending=False).head(top_n)
    
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
    plt.xlabel('Terms')
    plt.ylabel('IDF Value')
    plt.title('IDF Calculation Comparison')
    plt.xticks(x, df.index, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_idf_heatmap(corpus: List[List[str]], idf_dict: Dict[str, float], 
                    top_n: int = 20, figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Visualize where high-IDF terms appear in the corpus.
    
    Args:
        corpus: List of tokenized documents
        idf_dict: Dictionary with IDF values
        top_n: Number of top IDF terms to display
        figsize: Figure size (width, height)
    """
    # Sort terms by IDF value (highest first)
    sorted_terms = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
    high_idf_terms = [term for term, _ in sorted_terms[:top_n]]
    
    # Create a binary matrix of term presence
    presence_matrix = np.zeros((len(high_idf_terms), len(corpus)))
    
    for i, term in enumerate(high_idf_terms):
        for j, doc in enumerate(corpus):
            if term in doc:
                presence_matrix[i, j] = 1
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    # Use a custom colormap for binary data
    cmap = plt.cm.get_cmap('viridis', 2)
    
    # Plot heatmap
    sns.heatmap(presence_matrix, cmap=cmap, cbar=False, 
               yticklabels=high_idf_terms,
               xticklabels=[f'Doc {i}' for i in range(len(corpus))])
    
    # Add IDF values to y-axis labels
    ax = plt.gca()
    yticks = ax.get_yticklabels()
    
    # Create new tick labels with IDF values
    new_labels = []
    for i, term in enumerate(high_idf_terms):
        idf_val = idf_dict[term]
        new_labels.append(f"{term} ({idf_val:.2f})")
    
    # Apply new labels
    ax.set_yticklabels(new_labels)
    
    plt.title(f'Presence of High-IDF Terms Across Documents')
    plt.tight_layout()
    plt.show()


def plot_idf_term_importance(idf_dict: Dict[str, float], corpus: List[List[str]],
                           top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize term importance (IDF value vs. document frequency).
    
    Args:
        idf_dict: Dictionary with IDF values
        corpus: List of tokenized documents
        top_n: Number of terms to label
        figsize: Figure size (width, height)
    """
    # Count document frequency for each term
    doc_freq = defaultdict(int)
    for doc in corpus:
        # Count each term only once per document
        for term in set(doc):
            doc_freq[term] += 1
    
    # Convert to percentage
    num_docs = len(corpus)
    doc_freq_pct = {term: freq / num_docs * 100 for term, freq in doc_freq.items()}
    
    # Create scatter plot data
    terms = list(idf_dict.keys())
    x_values = [doc_freq_pct[term] for term in terms]
    y_values = [idf_dict[term] for term in terms]
    
    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(x_values, y_values, alpha=0.5)
    
    # Label important terms
    # Get the top terms by IDF value
    top_terms_by_idf = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for term, idf_val in top_terms_by_idf:
        x = doc_freq_pct[term]
        y = idf_val
        plt.text(x + 0.5, y, term, fontsize=9, ha='left', va='center')
    
    plt.xlabel('Document Frequency (%)')
    plt.ylabel('IDF Value')
    plt.title('Term Importance: IDF vs. Document Frequency')
    
    # Add IDF formula explanation
    plt.figtext(0.5, 0.01, 
               "IDF = log(N / df) where N = number of documents, df = document frequency", 
               ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the formula
    plt.show()


def plot_rare_vs_common_terms(idf_dict: Dict[str, float], corpus: List[List[str]],
                            n_terms: int = 10, figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Compare frequency of the rarest vs. most common terms.
    
    Args:
        idf_dict: Dictionary with IDF values
        corpus: List of tokenized documents
        n_terms: Number of terms to show for each category
        figsize: Figure size (width, height)
    """
    # Count document frequency for each term
    doc_freq = defaultdict(int)
    for doc in corpus:
        # Count each term only once per document
        for term in set(doc):
            doc_freq[term] += 1
    
    # Get the rarest terms (highest IDF)
    rare_terms = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)[:n_terms]
    
    # Get the most common terms (lowest IDF)
    common_terms = sorted(idf_dict.items(), key=lambda x: x[1])[:n_terms]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot rare terms
    rare_labels = [term for term, _ in rare_terms]
    rare_values = [doc_freq[term] for term, _ in rare_terms]
    rare_bars = ax1.barh(range(len(rare_labels)), rare_values, alpha=0.7)
    ax1.set_yticks(range(len(rare_labels)))
    ax1.set_yticklabels(rare_labels)
    ax1.invert_yaxis()  # Put the highest IDF at the top
    ax1.set_xlabel('Document Frequency (count)')
    ax1.set_title('Rarest Terms (Highest IDF)')
    
    # Add IDF values to labels
    for i, (term, idf) in enumerate(rare_terms):
        ax1.text(rare_values[i] + 0.1, i, f'IDF: {idf:.2f}', va='center')
    
    # Plot common terms
    common_labels = [term for term, _ in common_terms]
    common_values = [doc_freq[term] for term, _ in common_terms]
    common_bars = ax2.barh(range(len(common_labels)), common_values, alpha=0.7)
    ax2.set_yticks(range(len(common_labels)))
    ax2.set_yticklabels(common_labels)
    ax2.invert_yaxis()  # Put the lowest IDF at the top
    ax2.set_xlabel('Document Frequency (count)')
    ax2.set_title('Most Common Terms (Lowest IDF)')
    
    # Add IDF values to labels
    for i, (term, idf) in enumerate(common_terms):
        ax2.text(common_values[i] + 0.1, i, f'IDF: {idf:.2f}', va='center')
    
    plt.tight_layout()
    plt.show() 