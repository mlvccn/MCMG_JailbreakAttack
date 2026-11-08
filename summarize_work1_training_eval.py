import argparse
import json
import math
import re
from pathlib import Path


ITERATION_PATTERN = re.compile(r"evalaute_llama_iter_(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Work1 training-subset and independent-test ASR."
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--training-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def wilson_interval(successful: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successful / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [100 * (center - radius), 100 * (center + radius)]


def summarize_group(results: list[dict]) -> dict:
    successful = sum(bool(result.get("attack_success")) for result in results)
    total = len(results)
    return {
        "successful": successful,
        "total": total,
        "asr": successful / total if total else 0.0,
        "asr_percent": 100 * successful / total if total else 0.0,
        "wilson_95_percent": wilson_interval(successful, total),
    }


def iteration_from_path(path: Path) -> int:
    match = ITERATION_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected evaluation filename: {path.name}")
    return int(match.group(1))


def summarize_iteration(
    training_dir: Path,
    checkpoint_dir: Path,
    iteration: int,
) -> dict:
    cases_path = training_dir / f"test_cases_iter_{iteration}.jsonl"
    training_eval_path = training_dir / f"evalaute_llama_iter_{iteration}.json"
    independent_eval_path = checkpoint_dir / f"evalaute_llama_iter_{iteration}.json"
    for path in (cases_path, training_eval_path, independent_eval_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required result not found: {path}")

    split_by_id = {
        record["test_case_id"]: (record.get("metadata") or {}).get("mcts_split")
        for record in load_jsonl(cases_path)
    }
    training_report = load_json(training_eval_path)
    training_results = training_report.get("results", [])
    if set(split_by_id) != {
        result.get("test_case_id") for result in training_results
    }:
        raise ValueError(
            f"Training cases and evaluation IDs do not match at iteration {iteration}"
        )

    grouped = {
        split: [
            result
            for result in training_results
            if split_by_id[result["test_case_id"]] == split
        ]
        for split in ("search", "validation", "other")
    }
    independent_report = load_json(independent_eval_path)
    return {
        "iteration": iteration,
        "training_all": summarize_group(training_results),
        "mcts_search": summarize_group(grouped["search"]),
        "mcts_validation": summarize_group(grouped["validation"]),
        "training_other": summarize_group(grouped["other"]),
        "independent_test": summarize_group(independent_report.get("results", [])),
        "files": {
            "training_cases": str(cases_path),
            "training_evaluation": str(training_eval_path),
            "independent_evaluation": str(independent_eval_path),
        },
    }


def markdown_report(summary: dict) -> str:
    lines = [
        "# Work1 Training vs Independent ASR",
        "",
        "| Iteration | Training 66 | MCTS search 8 | MCTS validation 8 | Training other 50 | Independent test |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["iterations"]:
        values = []
        for key in (
            "training_all",
            "mcts_search",
            "mcts_validation",
            "training_other",
            "independent_test",
        ):
            group = item[key]
            values.append(
                f"{group['successful']}/{group['total']} "
                f"({group['asr_percent']:.1f}%)"
            )
        lines.append(f"| {item['iteration']} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "The MCTS search and validation groups are positions within the fixed 66-example training subset. The independent test results are read from the checkpoint directory and are not part of that training corpus.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    training_dir = args.training_dir or checkpoint_dir / "training_eval"
    evaluation_paths = sorted(
        training_dir.glob("evalaute_llama_iter_*.json"), key=iteration_from_path
    )
    if not evaluation_paths:
        raise FileNotFoundError(f"No training evaluations found in {training_dir}")

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "training_dir": str(training_dir),
        "iterations": [
            summarize_iteration(
                training_dir,
                checkpoint_dir,
                iteration_from_path(path),
            )
            for path in evaluation_paths
        ],
    }
    output_json = args.output_json or training_dir / "asr_comparison.json"
    output_md = args.output_md or training_dir / "asr_comparison.md"
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = markdown_report(summary)
    output_md.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved JSON report to: {output_json}")
    print(f"Saved Markdown report to: {output_md}")


if __name__ == "__main__":
    main()
