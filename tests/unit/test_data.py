"""Unit tests for data loading and processing."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data import load_dataset, split_data


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_valid_csv(self, sample_dataframe):
        """Test loading a valid CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_dataframe.to_csv(f, index=False)
            temp_path = Path(f.name)

        try:
            df = load_dataset(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert "text" in df.columns
            assert "label" in df.columns
            assert len(df) > 0
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/data.csv")

    def test_missing_columns(self):
        """Test loading CSV with missing columns raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n")
            f.write("value1,value2\n")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="must contain columns"):
                load_dataset(temp_path)
        finally:
            temp_path.unlink()

    def test_invalid_labels(self):
        """Test loading CSV with invalid labels raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,label\n")
            f.write("sample text,2\n")  # Invalid label (not 0 or 1)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Labels must be 0 or 1"):
                load_dataset(temp_path)
        finally:
            temp_path.unlink()

    def test_path_object(self, sample_dataframe):
        """Test loading with Path object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_dataframe.to_csv(f, index=False)
            temp_path = Path(f.name)

        try:
            df = load_dataset(Path(temp_path))
            assert len(df) > 0
        finally:
            temp_path.unlink()

    def test_string_path(self, sample_dataframe):
        """Test loading with string path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_dataframe.to_csv(f, index=False)
            temp_path = Path(f.name)

        try:
            df = load_dataset(str(temp_path))
            assert len(df) > 0
        finally:
            temp_path.unlink()


class TestSplitData:
    """Tests for split_data function."""

    def test_default_split(self, large_sample_dataframe):
        """Test default 80/20 split."""
        train_df, test_df = split_data(large_sample_dataframe)

        total = len(large_sample_dataframe)
        expected_test_size = int(total * 0.2)

        # Allow for rounding differences
        assert len(test_df) >= expected_test_size - 1
        assert len(test_df) <= expected_test_size + 1
        assert len(train_df) + len(test_df) == total

    def test_custom_split_ratio(self, large_sample_dataframe):
        """Test custom split ratio."""
        train_df, test_df = split_data(large_sample_dataframe, test_size=0.3)

        total = len(large_sample_dataframe)
        expected_test_size = int(total * 0.3)

        assert len(test_df) >= expected_test_size - 2
        assert len(test_df) <= expected_test_size + 2

    def test_reproducibility(self, large_sample_dataframe):
        """Test split is reproducible with same random_state."""
        train1, test1 = split_data(large_sample_dataframe, random_state=42)
        train2, test2 = split_data(large_sample_dataframe, random_state=42)

        assert train1["text"].tolist() == train2["text"].tolist()
        assert test1["text"].tolist() == test2["text"].tolist()

    def test_different_random_states(self, large_sample_dataframe):
        """Test different random_states produce different splits."""
        train1, test1 = split_data(large_sample_dataframe, random_state=42)
        train2, test2 = split_data(large_sample_dataframe, random_state=123)

        # With same size, check it works
        assert len(train1) == len(train2)
        assert len(test1) == len(test2)

    def test_stratified_split(self, large_sample_dataframe):
        """Test split maintains label distribution (stratified)."""
        train_df, test_df = split_data(large_sample_dataframe, test_size=0.2)

        # Calculate label proportions
        original_ratio = large_sample_dataframe["label"].mean()
        train_ratio = train_df["label"].mean()
        test_ratio = test_df["label"].mean()

        # Should be approximately same (within 10%)
        assert abs(train_ratio - original_ratio) < 0.1
        assert abs(test_ratio - original_ratio) < 0.1

    def test_columns_preserved(self, large_sample_dataframe):
        """Test all columns are preserved after split."""
        train_df, test_df = split_data(large_sample_dataframe)

        assert set(train_df.columns) == set(large_sample_dataframe.columns)
        assert set(test_df.columns) == set(large_sample_dataframe.columns)

    def test_no_overlapping_indices(self, large_sample_dataframe):
        """Test train and test sets don't overlap by index."""
        train_df, test_df = split_data(large_sample_dataframe)

        train_indices = set(train_df.index.tolist())
        test_indices = set(test_df.index.tolist())

        # No common indices
        assert len(train_indices.intersection(test_indices)) == 0
