#!/usr/bin/env python3
"""
CLI for sentiment MLOps training pipeline.

Usage:
    # Run training
    python -m pipelines.cli train
    python -m pipelines.cli train --champion
    python -m pipelines.cli train --data-path data/imdb_sample.csv

    # Show champion model info
    python -m pipelines.cli model-info
    python -m pipelines.cli model-info --model-name sentiment-classifier

    # Set model alias manually
    python -m pipelines.cli set-alias --model-name sentiment-classifier --version 1 --alias champion
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def train_command(args: argparse.Namespace) -> None:
    """Run training flow."""
    from pipelines.train import training_flow

    result = training_flow(
        data_path=args.data_path,
        register_model=args.register,
        set_champion=args.champion,
    )

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    print(f"Run ID: {result['run_id']}")
    print(f"Metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value:.4f}")

    if result["registered"]:
        print(f"Model version: {result['version']}")
        if result["is_champion"]:
            print("Status: Set as champion")
        elif args.champion:
            print("Status: Not promoted (current champion is better)")


def model_info_command(args: argparse.Namespace) -> None:
    """Show current champion model info."""
    from mlflow import MlflowClient

    from src.config import get_settings

    settings = get_settings()
    settings.configure_mlflow_environment()

    client = MlflowClient()

    try:
        version = client.get_model_version_by_alias(args.model_name, "champion")
        run = client.get_run(version.run_id)

        print("\n" + "=" * 50)
        print("Champion Model Info")
        print("=" * 50)
        print(f"Model: {args.model_name}")
        print(f"Version: {version.version}")
        print(f"Alias: champion")
        print(f"Run ID: {version.run_id}")
        print(f"Created: {version.creation_timestamp}")
        print(f"\nMetrics:")
        for key, value in sorted(run.data.metrics.items()):
            print(f"  {key}: {value:.4f}")
        print(f"\nParameters:")
        for key, value in sorted(run.data.params.items()):
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"No champion model found for '{args.model_name}': {e}")
        print("\nTo set a champion, run training with --champion flag:")
        print("  python -m pipelines.cli train --champion")
        sys.exit(1)


def set_alias_command(args: argparse.Namespace) -> None:
    """Manually set alias on model version."""
    from mlflow import MlflowClient

    from src.config import get_settings

    settings = get_settings()
    settings.configure_mlflow_environment()

    client = MlflowClient()

    try:
        # Verify version exists
        client.get_model_version(args.model_name, str(args.version))

        # Set alias
        client.set_registered_model_alias(
            args.model_name,
            args.alias,
            args.version,
        )
        print(f"Set alias '{args.alias}' on {args.model_name} v{args.version}")

    except Exception as e:
        print(f"Failed to set alias: {e}")
        sys.exit(1)


def list_versions_command(args: argparse.Namespace) -> None:
    """List all versions of a registered model."""
    from mlflow import MlflowClient

    from src.config import get_settings

    settings = get_settings()
    settings.configure_mlflow_environment()

    client = MlflowClient()

    try:
        # Get registered model
        model = client.get_registered_model(args.model_name)

        print("\n" + "=" * 50)
        print(f"Model: {args.model_name}")
        print("=" * 50)

        # Get aliases (model.aliases is dict[str, str] in MLflow 3.x)
        aliases = model.aliases if model.aliases else {}

        # List versions
        versions = client.search_model_versions(f"name='{args.model_name}'")
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

        if not versions:
            print("No versions found.")
            return

        print(f"\n{'Version':<10} {'Run ID':<36} {'Aliases':<20}")
        print("-" * 70)

        for v in versions:
            version_aliases = [
                alias for alias, ver in aliases.items() if ver == v.version
            ]
            alias_str = ", ".join(version_aliases) if version_aliases else "-"
            print(f"{v.version:<10} {v.run_id:<36} {alias_str:<20}")

    except Exception as e:
        print(f"Failed to list versions: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sentiment MLOps CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument(
        "--data-path",
        default="data/imdb_sample.csv",
        help="Path to dataset CSV (default: data/imdb_sample.csv)",
    )
    train_parser.add_argument(
        "--register",
        action="store_true",
        default=True,
        help="Register model in MLflow Registry (default: True)",
    )
    train_parser.add_argument(
        "--no-register",
        dest="register",
        action="store_false",
        help="Skip model registration",
    )
    train_parser.add_argument(
        "--champion",
        action="store_true",
        help="Set as champion if better than current",
    )
    train_parser.set_defaults(func=train_command)

    # model-info command
    info_parser = subparsers.add_parser("model-info", help="Show champion model info")
    info_parser.add_argument(
        "--model-name",
        default="sentiment-classifier",
        help="Model name (default: sentiment-classifier)",
    )
    info_parser.set_defaults(func=model_info_command)

    # set-alias command
    alias_parser = subparsers.add_parser("set-alias", help="Set model alias")
    alias_parser.add_argument(
        "--model-name",
        required=True,
        help="Registered model name",
    )
    alias_parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Model version number",
    )
    alias_parser.add_argument(
        "--alias",
        required=True,
        help="Alias to set (e.g., 'champion', 'challenger')",
    )
    alias_parser.set_defaults(func=set_alias_command)

    # list-versions command
    list_parser = subparsers.add_parser("list-versions", help="List model versions")
    list_parser.add_argument(
        "--model-name",
        default="sentiment-classifier",
        help="Model name (default: sentiment-classifier)",
    )
    list_parser.set_defaults(func=list_versions_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
