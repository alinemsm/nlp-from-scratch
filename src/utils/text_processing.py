"""
Text processing utilities for NLP tasks.

This module contains functions for text preprocessing and tokenization.
"""

from typing import List
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

def tokenize(
    text: str,
    ngram: int = 1,
    remove_stop_words: bool = False,
    lemmatizing: bool = False,
    stemming: bool = False
) -> List[str]:
    """
    Tokenize and preprocess text with various NLP techniques.

    This function performs text preprocessing including lowercasing, punctuation removal,
    stop word removal, lemmatization, and stemming. It can also generate n-grams.

    Parameters
    ----------
    text : str
        The input text to be tokenized and processed.
    ngram : int, default=1
        The size of n-grams to generate. If 1, returns individual tokens.
        If > 1, returns n-grams joined by underscores.
    remove_stop_words : bool, default=False
        Whether to remove English stop words from the tokens.
    lemmatizing : bool, default=False
        Whether to lemmatize the tokens to their base form.
    stemming : bool, default=False
        Whether to stem the tokens to their root form.

    Returns
    -------
    List[str]
        A list of processed tokens or n-grams.
    
    Examples
    --------
    >>> text = "The quick brown foxes are jumping over the lazy dogs."
    >>> tokenize(text, remove_stop_words=True)
    ['quick', 'brown', 'foxes', 'jumping', 'lazy', 'dogs']
    
    >>> tokenize(text, ngram=2)
    ['the_quick', 'quick_brown', 'brown_foxes', 'foxes_are', 'are_jumping',
     'jumping_over', 'over_the', 'the_lazy', 'lazy_dogs']
    
    >>> tokenize(text, lemmatizing=True)
    ['the', 'quick', 'brown', 'fox', 'be', 'jump', 'over', 'the', 'lazy', 'dog']

    Notes
    -----
    - The function processes text in the following order:
      1. Lowercasing
      2. Punctuation removal
      3. Tokenization
      4. Stop word removal (if enabled)
      5. Lemmatization (if enabled)
      6. Stemming (if enabled)
      7. N-gram generation (if ngram > 1)
    - It's recommended to use either lemmatization or stemming, but not both.
    - Requires NLTK data: 'stopwords', 'wordnet', and 'punkt'.
    """
    # Initialize NLTK components
    lemmatizer = WordNetLemmatizer() if lemmatizing else None
    stemmer = PorterStemmer() if stemming else None
    stop_words = set(stopwords.words('english')) if remove_stop_words else None

    # Text preprocessing
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Apply NLP techniques
    if remove_stop_words:
        tokens = [token for token in tokens if token not in stop_words]
    
    if lemmatizing:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    if stemming:
        tokens = [stemmer.stem(token) for token in tokens]

    # Generate n-grams if requested
    if ngram > 1:
        return ['_'.join(gram) for gram in nltk.ngrams(tokens, ngram)]
    
    return tokens 