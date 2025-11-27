"""Unit tests for model training and evaluation."""

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.model import (
    create_model,
    create_training_pipeline,
    evaluate_model,
    get_pipeline_params,
    train_model,
)


class TestCreateModel:
    """Tests for create_model function."""

    def test_default_parameters(self):
        """Test model created with default parameters."""
        model = create_model()

        assert isinstance(model, LogisticRegression)
        assert model.C == 1.0
        assert model.max_iter == 1000
        assert model.solver == "lbfgs"
        assert model.random_state == 42

    def test_custom_parameters(self):
        """Test model with custom parameters."""
        model = create_model(C=0.5, max_iter=500, solver="liblinear", random_state=123)

        assert model.C == 0.5
        assert model.max_iter == 500
        assert model.solver == "liblinear"
        assert model.random_state == 123


class TestCreateTrainingPipeline:
    """Tests for create_training_pipeline function."""

    def test_pipeline_creation(self):
        """Test pipeline is created with correct steps."""
        pipeline = create_training_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert "tfidf" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

    def test_pipeline_with_custom_params(self):
        """Test pipeline with custom vectorizer and model params."""
        vectorizer_params = {"max_features": 1000}
        model_params = {"C": 0.5}

        pipeline = create_training_pipeline(
            vectorizer_params=vectorizer_params,
            model_params=model_params,
        )

        assert pipeline.named_steps["tfidf"].max_features == 1000
        assert pipeline.named_steps["classifier"].C == 0.5


class TestTrainModel:
    """Tests for train_model function."""

    def test_training(self, large_sample_dataframe):
        """Test model training on sample data."""
        df = large_sample_dataframe
        X_train = df["text"].tolist()
        y_train = df["label"].tolist()

        pipeline = create_training_pipeline(
            vectorizer_params={"max_features": 100, "min_df": 1},
        )
        trained_pipeline = train_model(pipeline, X_train, y_train)

        # Check pipeline is fitted
        assert hasattr(trained_pipeline.named_steps["tfidf"], "vocabulary_")
        assert len(trained_pipeline.named_steps["tfidf"].vocabulary_) > 0

    def test_prediction_after_training(self, large_sample_dataframe):
        """Test predictions after training."""
        df = large_sample_dataframe
        X_train = df["text"].tolist()
        y_train = df["label"].tolist()

        pipeline = create_training_pipeline(
            vectorizer_params={"max_features": 100, "min_df": 1},
        )
        trained_pipeline = train_model(pipeline, X_train, y_train)

        # Test predictions
        predictions = trained_pipeline.predict(["This is great!", "This is terrible."])
        assert len(predictions) == 2
        assert all(p in [0, 1] for p in predictions)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_evaluation_metrics(self, large_sample_dataframe):
        """Test evaluation returns expected metrics."""
        df = large_sample_dataframe.sample(frac=1, random_state=42)  # Shuffle
        X = df["text"].tolist()
        y = df["label"].tolist()

        # Split manually - ensure both classes in both sets
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        pipeline = create_training_pipeline(
            vectorizer_params={"max_features": 100, "min_df": 1},
        )
        trained_pipeline = train_model(pipeline, X_train, y_train)

        metrics = evaluate_model(trained_pipeline, X_test, y_test)

        # Check all expected metrics are present
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

        # Check metrics are valid values
        for metric_value in metrics.values():
            assert 0.0 <= metric_value <= 1.0


class TestGetPipelineParams:
    """Tests for get_pipeline_params function."""

    def test_unfitted_pipeline(self):
        """Test params from unfitted pipeline."""
        pipeline = create_training_pipeline(
            vectorizer_params={"max_features": 500},
            model_params={"C": 0.5},
        )

        params = get_pipeline_params(pipeline)

        assert params["vectorizer_max_features"] == 500
        assert params["model_C"] == 0.5
        assert "model_max_iter" in params
        assert "model_solver" in params

    def test_fitted_pipeline(self, large_sample_dataframe):
        """Test params from fitted pipeline."""
        df = large_sample_dataframe
        X_train = df["text"].tolist()
        y_train = df["label"].tolist()

        pipeline = create_training_pipeline(
            vectorizer_params={"max_features": 100, "min_df": 1},
        )
        trained_pipeline = train_model(pipeline, X_train, y_train)

        params = get_pipeline_params(trained_pipeline)

        assert params["vectorizer_vocabulary_size"] > 0


class TestEndToEndTraining:
    """End-to-end training tests."""

    def test_full_training_workflow(self, large_sample_dataframe):
        """Test complete training workflow."""
        df = large_sample_dataframe.sample(frac=1, random_state=42)  # Shuffle

        # Split data
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df["text"].tolist()
        y_train = train_df["label"].tolist()
        X_test = test_df["text"].tolist()
        y_test = test_df["label"].tolist()

        # Create and train pipeline
        pipeline = create_training_pipeline(
            vectorizer_params={"max_features": 100, "min_df": 1},
            model_params={"C": 1.0, "max_iter": 500},
        )
        trained_pipeline = train_model(pipeline, X_train, y_train)

        # Evaluate
        metrics = evaluate_model(trained_pipeline, X_test, y_test)

        # Get params
        params = get_pipeline_params(trained_pipeline)

        # Verify results
        assert metrics["accuracy"] > 0.5  # Should be better than random
        assert params["vectorizer_max_features"] == 100
        assert params["model_C"] == 1.0

    def test_reproducibility(self, large_sample_dataframe):
        """Test training is reproducible with same random_state."""
        df = large_sample_dataframe
        X_train = df["text"].tolist()
        y_train = df["label"].tolist()

        # Train twice with same seed
        pipeline1 = create_training_pipeline(
            vectorizer_params={"max_features": 50, "min_df": 1},
            model_params={"random_state": 42},
        )
        train_model(pipeline1, X_train, y_train)

        pipeline2 = create_training_pipeline(
            vectorizer_params={"max_features": 50, "min_df": 1},
            model_params={"random_state": 42},
        )
        train_model(pipeline2, X_train, y_train)

        # Same predictions
        test_texts = ["Great product!", "Terrible quality!"]
        pred1 = pipeline1.predict(test_texts)
        pred2 = pipeline2.predict(test_texts)

        assert list(pred1) == list(pred2)
