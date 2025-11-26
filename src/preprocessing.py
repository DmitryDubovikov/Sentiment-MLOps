"""Text preprocessing utilities for sentiment classification."""

import logging
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text for sentiment analysis.

    Steps:
        1. Convert to lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Remove punctuation
        5. Remove extra whitespace

    Args:
        text: Raw input text

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def create_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    """
    Create TF-IDF vectorizer with specified parameters.

    Args:
        max_features: Maximum number of features
        ngram_range: Range of n-grams (min, max)
        min_df: Minimum document frequency
        max_df: Maximum document frequency (as fraction)

    Returns:
        Configured TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        preprocessor=clean_text,
        stop_words="english",
        lowercase=False,  # Already done in clean_text
    )

    logger.info(
        f"Created TF-IDF vectorizer: max_features={max_features}, "
        f"ngram_range={ngram_range}, min_df={min_df}, max_df={max_df}"
    )

    return vectorizer


def get_vectorizer_params(vectorizer: TfidfVectorizer) -> dict:
    """
    Extract vectorizer parameters for logging.

    Args:
        vectorizer: Fitted TfidfVectorizer

    Returns:
        Dictionary of vectorizer parameters
    """
    return {
        "vectorizer_max_features": vectorizer.max_features,
        "vectorizer_ngram_range": str(vectorizer.ngram_range),
        "vectorizer_min_df": vectorizer.min_df,
        "vectorizer_max_df": vectorizer.max_df,
        "vectorizer_vocabulary_size": len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else 0,
    }


def create_preprocessing_pipeline(vectorizer: TfidfVectorizer) -> Pipeline:
    """
    Create sklearn Pipeline with text preprocessing.

    Note: The vectorizer already includes clean_text as preprocessor,
    so this pipeline just wraps the vectorizer for consistency.

    Args:
        vectorizer: Configured TfidfVectorizer

    Returns:
        sklearn Pipeline
    """
    return Pipeline([("tfidf", vectorizer)])
