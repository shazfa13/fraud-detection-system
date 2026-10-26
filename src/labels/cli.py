"""Maintenance commands for the Stage 2 prediction registry."""
from __future__ import annotations

import argparse
import json

from src.labels import config
from src.labels.store import get_stats, initialize, replay_prediction_jsonl
from src.streaming import config as streaming_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Initialize the Stage 2 SQLite database")
    replay = sub.add_parser("replay", help="Idempotently import Stage 1 prediction JSONL")
    replay.add_argument("--jsonl", default=str(streaming_config.STAGE1_OUTPUT_PATH))
    sub.add_parser("stats", help="Print Stage 2 counts")
    args = parser.parse_args()
    if args.command == "init":
        initialize()
        result = {"database": str(config.STAGE2_DB_PATH), "initialized": True}
    elif args.command == "replay":
        result = replay_prediction_jsonl(args.jsonl)
    else:
        result = get_stats()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
