import argparse
import json
import re
from pathlib import Path

import torch
from torchvision.utils import save_image


CHECKPOINT_PATTERN = re.compile(r"checkpoint_iter_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Work1 checkpoints into per-iteration test case files."
    )
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        type=Path,
        default=Path("output/test_cases/work1_qwen2_5_vl"),
        help="Directory containing checkpoint_iter_*.pt and test_cases.jsonl.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Template test_cases.jsonl (default: CHECKPOINT_DIR/test_cases.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: CHECKPOINT_DIR).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)

    if not records:
        raise ValueError(f"No test cases found in {path}")
    return records


def checkpoint_iteration(path: Path) -> int:
    match = CHECKPOINT_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint filename: {path.name}")
    return int(match.group(1))


def load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    required = {"iteration", "adv_suffix", "adv_img_prompt"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"{path} is missing checkpoint fields: {sorted(missing)}")
    if not isinstance(checkpoint["adv_suffix"], str):
        raise ValueError(f"{path}: adv_suffix must be a string")
    if not torch.is_tensor(checkpoint["adv_img_prompt"]):
        raise ValueError(f"{path}: adv_img_prompt must be a tensor")
    return checkpoint


def make_iteration_record(
    template: dict,
    iteration: int,
    suffix: str,
    image_path: str,
) -> dict:
    record = dict(template)
    metadata = dict(record.get("metadata") or {})
    original_prompt = metadata.get("original_prompt", record.get("original_prompt"))
    if not isinstance(original_prompt, str):
        raise ValueError(
            f"Test case {record.get('test_case_id', '<unknown>')} has no original_prompt"
        )

    jailbreak_prompt = original_prompt + suffix
    record["image_path"] = image_path
    record["prompt"] = jailbreak_prompt
    metadata["jailbreak_prompt"] = jailbreak_prompt
    metadata["jailbreak_image_path"] = image_path
    metadata["checkpoint_iteration"] = iteration
    metadata["adv_suffix"] = suffix
    record["metadata"] = metadata
    return record


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    template_path = args.template or checkpoint_dir / "test_cases.jsonl"
    output_dir = args.output_dir or checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = load_jsonl(template_path)
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_iter_*.pt"), key=checkpoint_iteration
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_iter_*.pt files found in {checkpoint_dir}")

    for checkpoint_path in checkpoints:
        checkpoint = load_checkpoint(checkpoint_path)
        iteration = checkpoint_iteration(checkpoint_path)
        saved_iteration = int(checkpoint["iteration"])
        if saved_iteration != iteration:
            raise ValueError(
                f"Iteration mismatch in {checkpoint_path}: "
                f"filename={iteration}, checkpoint={saved_iteration}"
            )

        suffix = checkpoint["adv_suffix"]
        image = checkpoint["adv_img_prompt"].detach().float().cpu()
        image_path = output_dir / f"work1_universal_iter_{iteration}.png"
        save_image(image.clamp(0, 1), image_path)

        try:
            stored_image_path = image_path.resolve().relative_to(
                Path.cwd().resolve()
            ).as_posix()
        except ValueError:
            stored_image_path = image_path.resolve().as_posix()

        records = [
            make_iteration_record(
                template, iteration, suffix, stored_image_path
            )
            for template in templates
        ]
        cases_path = output_dir / f"test_cases_iter_{iteration}.jsonl"
        write_jsonl(cases_path, records)
        print(
            f"Iteration {iteration}: wrote {len(records)} cases to {cases_path} "
            f"and image to {image_path}"
        )


if __name__ == "__main__":
    main()
