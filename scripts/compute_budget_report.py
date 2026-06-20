"""Standalone compute-budget report for POEM candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.build import build_model, count_parameters
from models.config import config_for_model_type
from training.compute_budget import DEFAULT_TRAIN_EPOCHS, format_budget_plan, plan_epoch_budget
from training.data import count_tokenized_events, discover_motif_files, split_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--include_long_motifs", action="store_true")
    parser.add_argument("--model_type", choices=["A", "B", "C", "D", "E", "F"], default=None)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    files = discover_motif_files(args.data_dir, include_long_motifs=args.include_long_motifs)
    train_files, _ = split_files(files, smoke_test=args.smoke_test)
    probe_config = config_for_model_type(args.model_type or "D", smoke_test=args.smoke_test)
    print(f"Counting tokenized events for {len(train_files)} training files...")
    dataset_tokens = count_tokenized_events(train_files, max_seq_len=probe_config.max_seq_len)
    model_types = [args.model_type] if args.model_type else ["A", "B", "C", "D", "E", "F"]
    for model_type in model_types:
        config = config_for_model_type(model_type, smoke_test=args.smoke_test)
        model = build_model(config)
        plan = plan_epoch_budget(dataset_tokens, count_parameters(model), selected_epochs=args.epochs)
        print(f"\nPOEM candidate {model_type}")
        print(format_budget_plan(plan))


if __name__ == "__main__":
    main()
