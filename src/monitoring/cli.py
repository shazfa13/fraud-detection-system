"""Run one read-only Stage 3 monitoring evaluation and persist its report and signals."""
import argparse
import json

from src.monitoring.monitor import run_monitor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", choices=("V1", "V2"))
    args = parser.parse_args()
    print(json.dumps(run_monitor(model_version=args.model_version), indent=2))


if __name__ == "__main__":
    main()
