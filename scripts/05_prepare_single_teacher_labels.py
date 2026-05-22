from __future__ import annotations

import argparse

from scripts._lib.repro_io import load_yaml, project_root
from scripts._lib.step05_single_teacher_labels import (
    build_single_teacher_labels,
    resolve_single_teacher_label_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare self-confidence filtered single-teacher labels.")
    parser.add_argument(
        "--config",
        default="scripts/config/reproduction_default.yaml",
        help="Path to reproduction YAML config.",
    )
    parser.add_argument(
        "--pattern-limit",
        type=int,
        default=1,
        help="Single-teacher mode expects one pattern.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(root / args.config)
    resolved = resolve_single_teacher_label_paths(root, config)
    build_single_teacher_labels(
        root=root,
        resolved_paths=resolved,
        config=config,
        pattern_limit=args.pattern_limit,
    )


if __name__ == "__main__":
    main()
