import argparse
import json
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qwen2.5-VL, Qwen3-VL, or GLM-4.1V responses with vLLM."
    )
    parser.add_argument(
        "--input_jsonl", type=Path, action="append", required=True,
        help="Input JSONL; repeat together with --output_jsonl for multiple files.",
    )
    parser.add_argument(
        "--output_jsonl", type=Path, action="append", required=True,
        help="Output JSONL; must have the same count as --input_jsonl.",
    )
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--model_name", default="qwen2_5_vl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--text_only", action="store_true",
        help="Send only ordinary text, without images or appended suffixes.",
    )
    parser.add_argument(
        "--text_field", default="original_prompt",
        help="Text-only input field, at top level or in metadata (default: original_prompt).",
    )
    return parser.parse_args()


def get_text_prompt(case: dict[str, Any], field: str = "original_prompt") -> str:
    value = case.get(field)
    if value is None:
        metadata = case.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Text-only input requires a nonempty string {field!r} at top level or in metadata. "
            "No automatic fallback to prompt is used, because it may contain an attack suffix. "
            "For clean prompt-only files, explicitly use --text_field prompt."
        )
    return value


def load_test_cases(
    path: Path, text_only: bool = False, text_field: str = "original_prompt",
) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

            if not isinstance(case, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            required = {"test_case_id"} if text_only else {"test_case_id", "image_path", "prompt"}
            missing = required.difference(case)
            if missing:
                raise ValueError(
                    f"{path}:{line_number} is missing fields: {sorted(missing)}"
                )
            if text_only:
                try:
                    get_text_prompt(case, text_field)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
            cases.append(case)

    if not cases:
        raise ValueError(f"No test cases found in {path}")
    return cases


def load_image(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def make_vllm_input(
    processor: Any,
    image: Image.Image,
    prompt: str,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": prompt_text,
        "multi_modal_data": {"image": image},
    }


def make_vllm_text_input(processor: Any, prompt: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return {
        "prompt": processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        ),
    }


def make_result(
    test_case: dict[str, Any],
    response: str,
    model_name: str,
    text_prompt: str | None = None,
) -> dict[str, Any]:
    metadata = dict(test_case.get("metadata") or {})
    metadata.pop("original_image_path", None)
    if "jailbreak_image_path" in metadata:
        metadata["image_path"] = metadata.pop("jailbreak_image_path")

    if text_prompt is not None:
        for key in ("image_path", "jailbreak_prompt", "jailbreak_image_path",
                    "adv_suffix", "checkpoint_iteration"):
            metadata.pop(key, None)
        metadata.update({
            "original_prompt": text_prompt,
            "attack_method": "none",
            "input_mode": "text_only",
        })

    result = {
        "test_case_id": test_case["test_case_id"],
        "model_response": response,
        "model_name": model_name,
        "metadata": metadata,
    }
    if text_prompt is not None:
        result["original_prompt"] = text_prompt
    return result


def append_results(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        file.flush()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if len(args.input_jsonl) != len(args.output_jsonl):
        raise ValueError(
            "--input_jsonl and --output_jsonl must be provided the same number of times"
        )
    input_paths = {path.resolve() for path in args.input_jsonl}
    if any(path.resolve() in input_paths for path in args.output_jsonl):
        raise ValueError("Output paths must not overwrite input files")
    inputs = [load_test_cases(path, args.text_only, args.text_field) for path in args.input_jsonl]

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    print(f"Loading processor: {args.model_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": 0, "video": 0} if args.text_only else {"image": 1},
        "trust_remote_code": True,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.seed is not None:
        llm_kwargs["seed"] = args.seed

    print(f"Loading vLLM model: {args.model_path}")
    model = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    for input_path, output_path, test_cases in zip(args.input_jsonl, args.output_jsonl, inputs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generation_record = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_jsonl': str(input_path),
            'output_jsonl': str(output_path),
            'batch_size': args.batch_size,
            'model_name': args.model_name,
            'text_only': args.text_only,
            'text_field': args.text_field if args.text_only else None,
            'seed': args.seed,
            'case_count': len(test_cases),
            'engine_arguments': llm_kwargs,
            'sampling_parameters': repr(sampling_params),
            'temperature': sampling_params.temperature,
            'max_tokens': sampling_params.max_tokens,
            'chat_template': processor.chat_template,
            'versions': {name: version(name) for name in ('vllm', 'transformers', 'torch')},
        }
        output_path.with_suffix('.generation_config.json').write_text(
            json.dumps(generation_record, ensure_ascii=False, indent=2), encoding='utf-8',
        )
        output_path.write_text("", encoding="utf-8")
        total = len(test_cases)
        print(f"Generating {total} cases from: {input_path}")

        for start in range(0, total, args.batch_size):
            batch = test_cases[start : start + args.batch_size]
            if args.text_only:
                images = []
                requests = [make_vllm_text_input(processor, get_text_prompt(case, args.text_field))
                            for case in batch]
            else:
                images = [load_image(case["image_path"]) for case in batch]
                requests = [
                    make_vllm_input(processor, image, case["prompt"])
                    for case, image in zip(batch, images)
                ]
            outputs = model.generate(requests, sampling_params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(batch)} inputs"
                )

            results = [
                make_result(
                    case,
                    output.outputs[0].text.strip(),
                    args.model_name,
                    text_prompt=get_text_prompt(case, args.text_field) if args.text_only else None,
                )
                for case, output in zip(batch, outputs)
            ]
            append_results(output_path, results)
            for image in images:
                image.close()
            print(f"Generated {min(start + len(batch), total)}/{total}")

        print(f"Saved responses to: {output_path}")


if __name__ == "__main__":
    main()
