"""Unit tests for text preprocessing utilities."""

import pytest

from src.preprocessing import (
    clean_text,
    create_preprocessing_pipeline,
    create_vectorizer,
    get_vectorizer_params,
)


class TestCleanText:
    """Tests for clean_text function."""

    def test_lowercase(self):
        """Test text is converted to lowercase."""
        assert clean_text("HELLO WORLD") == "hello world"

    def test_remove_html_tags(self):
        """Test HTML tags are removed."""
        text = "<p>Hello <b>world</b></p>"
        result = clean_text(text)
        assert "<" not in result
        assert ">" not in result
        assert "hello" in result
        assert "world" in result

    def test_remove_urls(self):
        """Test URLs are removed."""
        text = "Check out https://example.com and www.test.org"
        result = clean_text(text)
        assert "https" not in result
        assert "www" not in result
        assert "check out" in result

    def test_remove_punctuation(self):
        """Test punctuation is removed."""
        text = "Hello, world! How are you?"
        result = clean_text(text)
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_remove_extra_whitespace(self):
        """Test extra whitespace is normalized."""
        text = "Hello    world   test"
        result = clean_text(text)
        assert result == "hello world test"

    def test_empty_string(self):
        """Test empty string returns empty string."""
        assert clean_text("") == ""

    def test_non_string_input(self):
        """Test non-string input returns empty string."""
        assert clean_text(None) == ""
        assert clean_text(123) == ""
        assert clean_text([]) == ""

    def test_complex_text(self):
        """Test complex text with multiple elements."""
        text = "<p>Check out https://example.com! It's AMAZING!!!</p>"
        result = clean_text(text)
        assert "https" not in result
        assert "AMAZING" not in result
        assert "amazing" in result
        assert "<" not in result


class TestCreateVectorizer:
    """Tests for create_vectorizer function."""

    def test_default_parameters(self):
        """Test vectorizer created with default parameters."""
        vectorizer = create_vectorizer()
        assert vectorizer.max_features == 5000
        assert vectorizer.ngram_range == (1, 2)
        assert vectorizer.min_df == 2
        assert vectorizer.max_df == 0.95

    def test_custom_parameters(self):
        """Test vectorizer with custom parameters."""
        vectorizer = create_vectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.8,
        )
        assert vectorizer.max_features == 1000
        assert vectorizer.ngram_range == (1, 3)
        assert vectorizer.min_df == 5
        assert vectorizer.max_df == 0.8

    def test_preprocessor_set(self):
        """Test that clean_text is set as preprocessor."""
        vectorizer = create_vectorizer()
        assert vectorizer.preprocessor is not None
        assert callable(vectorizer.preprocessor)

    def test_stop_words_english(self):
        """Test English stop words are set."""
        vectorizer = create_vectorizer()
        assert vectorizer.stop_words == "english"


class TestGetVectorizerParams:
    """Tests for get_vectorizer_params function."""

    def test_unfitted_vectorizer(self):
        """Test params extraction from unfitted vectorizer."""
        vectorizer = create_vectorizer(max_features=1000)
        params = get_vectorizer_params(vectorizer)

        assert params["vectorizer_max_features"] == 1000
        assert params["vectorizer_ngram_range"] == "(1, 2)"
        assert params["vectorizer_vocabulary_size"] == 0

    def test_fitted_vectorizer(self):
        """Test params extraction from fitted vectorizer."""
        vectorizer = create_vectorizer(max_features=100, min_df=1)
        texts = ["hello world", "test document", "another test"]
        vectorizer.fit(texts)

        params = get_vectorizer_params(vectorizer)

        assert params["vectorizer_max_features"] == 100
        assert params["vectorizer_vocabulary_size"] > 0


class TestCreatePreprocessingPipeline:
    """Tests for create_preprocessing_pipeline function."""

    def test_pipeline_creation(self):
        """Test preprocessing pipeline is created correctly."""
        vectorizer = create_vectorizer()
        pipeline = create_preprocessing_pipeline(vectorizer)

        assert "tfidf" in pipeline.named_steps
        assert pipeline.named_steps["tfidf"] is vectorizer

    def test_pipeline_fit_transform(self):
        """Test pipeline can fit and transform text."""
        vectorizer = create_vectorizer(max_features=100, min_df=1)
        pipeline = create_preprocessing_pipeline(vectorizer)

        texts = ["hello world", "test document", "another test"]
        result = pipeline.fit_transform(texts)

        assert result.shape[0] == 3  # Number of documents
        assert result.shape[1] > 0  # Has features
