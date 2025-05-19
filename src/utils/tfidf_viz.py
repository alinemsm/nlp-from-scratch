"""
Visualization utilities for NLP analysis.

This module contains functions for visualizing text analysis results,
particularly for TF-IDF and other text vectorization outputs.
"""

from typing import List, Dict
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_top_tfidf_terms(tfidf_corpus, top_n=5, cmap="YlGnBu", annot=True, fmt=".2f"):
    """
    Create and display a heatmap of the top TF-IDF terms for each document.
    
    Parameters:
    -----------
    tfidf_corpus : list of dict
        A list where each element is a dictionary mapping terms to their TF-IDF scores
    top_n : int, default=5
        Number of top terms to display for each document
    cmap : str, default="YlGnBu"
        Color map for the heatmap
    annot : bool, default=True
        Whether to annotate cells with values
    fmt : str, default=".2f"
        Format string for annotations
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the heatmap
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    # Create a dictionary to store top terms and their scores for each document
    top_terms_per_doc = defaultdict(dict)
    
    # For each document, find the top n terms by TF-IDF score
    for i, tfidf_doc in enumerate(tfidf_corpus):
        top_terms = sorted(tfidf_doc.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for term, score in top_terms:
            top_terms_per_doc[i][term] = score
    
    # Convert to DataFrame and fill missing values with 0
    top_terms_df = pd.DataFrame(top_terms_per_doc).fillna(0)
    
    # Create and display the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(top_terms_df, cmap=cmap, annot=annot, fmt=fmt)
    plt.title("Top TF-IDF Terms per Document")
    plt.xlabel("Document Index")
    plt.ylabel("Terms")
    plt.tight_layout()
    
    # fig = plt.gcf()
    plt.show()
    
    # return fig