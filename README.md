# NLP From Scratch

This repository is intended to contain code that explores the basics of NLP from first principles. It focuses on understanding rather than efficiency.

## Project Structure

```
├── data/               # Data files
│   ├── raw/           # Original, immutable data
│   └── processed/     # Cleaned and processed data
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── src/              # Source code
│   ├── data/         # Scripts for data processing
│   ├── features/     # Scripts for feature engineering
│   │   └── vectorization.py  # Text vectorization utilities (TF-IDF)
│   ├── models/       # Scripts for model training and prediction
│   └── utils/        # Utility functions and helper scripts
├── tests/            # Unit tests
│   └── test_vectorization.py  # Tests for vectorization functions
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Features

### Text Vectorization

The `src/features/vectorization.py` module provides functions for text vectorization:

- Term Frequency (TF) calculation
- Inverse Document Frequency (IDF) calculation 
- TF-IDF computation
- Vectorized implementations for improved performance

Examples of usage can be found in the docstrings and the `notebooks/tf-idf.ipynb` notebook.

## Running Tests

The project includes a test suite for the vectorization functionality:

```bash
# Run all tests
python -m pytest

# Run vectorization tests
python -m pytest tests/test_vectorization.py

# Run tests with verbose output
python -m pytest -v
```

