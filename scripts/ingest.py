import argparse
from pathlib import Path
from credit_risk.pipelines.ingest import ingest


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Ingest Freddie Mac loan-level data.")

    parser.add_argument(
        "--year",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    ingest(
        year=args.year,
        raw_root=Path("data/01_raw/freddie_mac"),
        interim_root=Path("data/02_interim"),
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
