#!/usr/bin/env python3
"""Generate Work1 universal attack checkpoints and independent test cases."""

import argparse
import json
from pathlib import Path

import yaml

from attacks.work1.attack import Work1Attack


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/work1_qwen2_5_vl.yaml"),
    )
    parser.add_argument(
        "--behaviors-file",
        type=Path,
        default=Path("dataset/harmbench_stand200.json"),
    )
    return parser.parse_args()


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{path} must contain a parameters mapping")
    return config, parameters


def load_behaviors(path):
    behaviors = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(behaviors, list) or not behaviors:
        raise ValueError(f"{path} must contain a nonempty JSON array")
    required = {"id", "original_prompt", "image_path"}
    for index, behavior in enumerate(behaviors):
        if not isinstance(behavior, dict):
            raise ValueError(f"Behavior {index} is not an object")
        missing = required.difference(behavior)
        if missing:
            raise ValueError(f"Behavior {index} is missing fields: {sorted(missing)}")
    return behaviors


def write_jsonl(path, cases):
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    raw_config, parameters = load_config(args.config)
    behaviors = load_behaviors(args.behaviors_file)
    save_dir = Path(parameters["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(raw_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    attack = Work1Attack(
        {**parameters, "_registry_name": "work1"},
        output_image_dir=str(save_dir),
    )
    cases = [
        attack.generate_test_case(
            original_prompt=behavior["original_prompt"],
            image_path=behavior["image_path"],
            case_id=behavior["id"],
        )
        for behavior in behaviors
    ]
    output_path = save_dir / "test_cases.jsonl"
    write_jsonl(output_path, cases)
    print(f"Saved {len(cases)} Work1 test cases to {output_path}")


if __name__ == "__main__":
    main()
