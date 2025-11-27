"""Custom exceptions for FastAPI application."""


class ModelNotLoadedError(Exception):
    """Raised when attempting to use a model that hasn't been loaded."""

    pass


class ModelLoadError(Exception):
    """Raised when model loading fails."""

    pass


class ChampionNotFoundError(Exception):
    """Raised when no champion model is found in the registry."""

    pass
