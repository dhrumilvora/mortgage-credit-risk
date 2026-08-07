import argparse
from pathlib import Path

from credit_risk.pipelines.ingest import ingest
from credit_risk.utils.config import read_config


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Ingest Freddie Mac loan-level data.")

    parser.add_argument(
        "--project-path",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing config/.",
    )

    parser.add_argument(
        "--year",
        type=int,
        help="Override the configured Freddie Mac vintage.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        help="Override the configured performance-file chunk size.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    config = read_config(args.project_path)
    data_config = config["parameters"]["data"]

    if args.year is not None:
        data_config["vintage"] = args.year

    if args.chunksize is not None:
        data_config["ingestion"]["chunksize"] = args.chunksize

    ingest(config)


if __name__ == "__main__":
    main()
