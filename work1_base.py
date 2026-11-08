"""Minimal data and component contracts required by the Work1 release."""

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict


class TestCase:
    def __init__(self, test_case_id, prompt, image_path, metadata=None):
        self.test_case_id = str(test_case_id)
        self.prompt = prompt
        self.image_path = str(image_path)
        self.metadata = dict(metadata or {})

    def to_dict(self):
        return {
            "test_case_id": self.test_case_id,
            "image_path": self.image_path,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }


class BaseAttack:
    CONFIG_CLASS = None

    def __init__(self, config: Dict[str, Any] = None, output_image_dir=None):
        self.config = dict(config or {})
        self.name = self.__class__.__name__
        self.registry_name = self.config.get("_registry_name")
        self.output_image_dir = Path(output_image_dir) if output_image_dir else None
        self._process_config(self.config)

    def _process_config(self, config):
        if self.CONFIG_CLASS is not None and is_dataclass(self.CONFIG_CLASS):
            allowed = {field.name for field in fields(self.CONFIG_CLASS)}
            self.cfg = self.CONFIG_CLASS(
                **{key: value for key, value in config.items() if key in allowed}
            )
            for field in fields(self.cfg):
                if field.default is MISSING and field.default_factory is MISSING:
                    if getattr(self.cfg, field.name, None) is None:
                        raise ValueError(f"Missing required configuration field: {field.name}")
        else:
            self.cfg = config

    def create_test_case(
        self,
        case_id,
        jailbreak_prompt,
        jailbreak_image_path,
        original_prompt,
        original_image_path=None,
        metadata=None,
        **kwargs,
    ):
        attack_method = self.registry_name or self.__class__.__name__.replace(
            "Attack", ""
        ).lower()
        case_metadata = {
            "attack_method": attack_method,
            "original_prompt": original_prompt,
            "jailbreak_prompt": jailbreak_prompt,
            "jailbreak_image_path": str(jailbreak_image_path),
        }
        if original_image_path:
            case_metadata["original_image_path"] = str(original_image_path)
        case_metadata.update(metadata or {})
        case_metadata.update(kwargs)
        return TestCase(
            test_case_id=case_id,
            prompt=jailbreak_prompt,
            image_path=jailbreak_image_path,
            metadata=case_metadata,
        )
