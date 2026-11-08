import argparse
import csv
import json
import re
from pathlib import Path

import torch


CHECKPOINT_PATTERN = re.compile(r"checkpoint_iter_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build formal generation cases for a Work1 training subset."
    )
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--selection-file", type=Path, default=None)
    parser.add_argument("--iterations", type=int, nargs="*", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required = {"iteration", "adv_suffix", "attack_config"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return checkpoint


def load_corpus(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Invalid Work1 corpus: {path}")
    records = [(row[0], row[1]) for row in rows[1:] if len(row) >= 2]
    if not records:
        raise ValueError(f"No behavior rows found in {path}")
    return records


def normalized_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def validate_subset_indices(indices: list[int], population_size: int) -> None:
    if not indices:
        raise ValueError("Training selection is empty")
    if len(indices) != len(set(indices)):
        raise ValueError("Training selection contains duplicate corpus indices")
    invalid = [index for index in indices if not 0 <= index < population_size]
    if invalid:
        raise ValueError(f"Training selection contains invalid indices: {invalid}")


def build_records(
    corpus: list[tuple[str, str]],
    selected_indices: list[int],
    search_indices: list[int],
    validation_indices: list[int],
    suffix: str,
    image_path: str,
    iteration: int,
) -> list[dict]:
    selected_size = len(selected_indices)
    search_set = set(search_indices)
    validation_set = set(validation_indices)
    invalid = sorted(
        index
        for index in search_set | validation_set
        if not 0 <= index < selected_size
    )
    if invalid:
        raise ValueError(f"MCTS subset positions are out of range: {invalid}")
    if search_set & validation_set:
        raise ValueError("MCTS search and validation subsets overlap")

    records = []
    for subset_position, corpus_index in enumerate(selected_indices):
        behavior, _target = corpus[corpus_index]
        if subset_position in search_set:
            split = "search"
        elif subset_position in validation_set:
            split = "validation"
        else:
            split = "other"
        jailbreak_prompt = behavior + suffix
        records.append(
            {
                "test_case_id": (
                    f"work1_train_{subset_position:02d}_corpus_{corpus_index:04d}"
                ),
                "image_path": image_path,
                "prompt": jailbreak_prompt,
                "metadata": {
                    "attack_method": "work1",
                    "original_prompt": behavior,
                    "jailbreak_prompt": jailbreak_prompt,
                    "checkpoint_iteration": iteration,
                    "adv_suffix": suffix,
                    "training_subset_position": subset_position,
                    "corpus_index": corpus_index,
                    "mcts_split": split,
                    "image_path": image_path,
                },
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def checkpoint_iteration(path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir or checkpoint_dir / "training_eval"
    selection_path = args.selection_file or checkpoint_dir / "selected_training_indices.json"
    selection = load_json(selection_path)
    selected_indices = [int(index) for index in selection.get("selected_indices", [])]

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_iter_*.pt"), key=checkpoint_iteration
    )
    if args.iterations:
        wanted = set(args.iterations)
        checkpoints = [p for p in checkpoints if checkpoint_iteration(p) in wanted]
    if not checkpoints:
        raise FileNotFoundError(f"No matching checkpoints found in {checkpoint_dir}")

    first_checkpoint = load_checkpoint(checkpoints[0])
    configured_corpus = first_checkpoint["attack_config"].get("corpus")
    corpus_path = args.corpus or Path(configured_corpus or selection.get("corpus", ""))
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {corpus_path}")
    corpus = load_corpus(corpus_path)
    validate_subset_indices(selected_indices, len(corpus))

    manifest = {
        "checkpoint_dir": normalized_path(checkpoint_dir),
        "corpus": normalized_path(corpus_path),
        "selection_file": normalized_path(selection_path),
        "selected_indices": selected_indices,
        "iterations": [],
    }

    for checkpoint_path in checkpoints:
        checkpoint = load_checkpoint(checkpoint_path)
        iteration = checkpoint_iteration(checkpoint_path)
        if int(checkpoint["iteration"]) != iteration:
            raise ValueError(f"Iteration mismatch in {checkpoint_path}")
        search_indices = [
            int(index)
            for index in checkpoint.get("mcts_indices_in_training_subset", [])
        ]
        validation_indices = [
            int(index)
            for index in checkpoint.get(
                "mcts_validation_indices_in_training_subset", []
            )
        ]
        if not search_indices or not validation_indices:
            raise ValueError(
                f"{checkpoint_path} does not contain both MCTS subset manifests"
            )

        image_path = checkpoint_dir / f"work1_universal_iter_{iteration}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Checkpoint image not found: {image_path}")
        stored_image_path = normalized_path(image_path)
        suffix = checkpoint["adv_suffix"]
        records = build_records(
            corpus,
            selected_indices,
            search_indices,
            validation_indices,
            suffix,
            stored_image_path,
            iteration,
        )
        output_path = output_dir / f"test_cases_iter_{iteration}.jsonl"
        write_jsonl(output_path, records)
        manifest["iterations"].append(
            {
                "iteration": iteration,
                "checkpoint": normalized_path(checkpoint_path),
                "test_cases": normalized_path(output_path),
                "search_indices": search_indices,
                "validation_indices": validation_indices,
                "counts": {
                    split: sum(
                        record["metadata"]["mcts_split"] == split
                        for record in records
                    )
                    for split in ("search", "validation", "other")
                },
            }
        )
        print(f"Iteration {iteration}: wrote {len(records)} cases to {output_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "training_eval_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
