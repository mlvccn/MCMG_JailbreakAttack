from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Union

import csv
import json
import random
from torchvision.utils import save_image
from PIL import Image

from dataclasses import dataclass
from work1_base import BaseAttack, TestCase


def select_training_indices(population_size: int, sample_size: int, seed: int):
    """Select a reproducible subset without changing global RNG state."""
    if population_size < sample_size:
        raise ValueError(
            f"Training corpus contains {population_size} samples, but "
            f"training_sample_size={sample_size}"
        )
    return random.Random(seed).sample(range(population_size), sample_size)


@dataclass
class Work1Config:
    n_iters: int = 2000
    eps: int = 32
    alpha: float = 1.0
    corpus: str = "attacks/data/harmful_behaviors.csv"
    prior_image: str = ""
    batch_size: int = 8
    target_model: str = None
    model_path: Optional[str] = None
    model_device: Optional[str] = None
    model_dtype: str = 'bfloat16'
    model_image_size: int = 448
    model_max_txt_len: int = 2048
    model_system_prompt: Optional[str] = None
    model_attn_implementation: str = 'sdpa'
    image_gradient_sub_batch_size: int = 2
    preview_max_new_tokens: int = 300
    save_dir: str = "output/test_cases/work1"
    random_seed: int = 42
    training_sample_size: int = 66

    use_mcts: bool = True
    mcts_min_accept_depth: int = 15
    mcts_deep_accept_improvement: float = 0.02
    mcts_shallow_accept_improvement: float = 0.05
    mcts_position_stride: int = 5
    mcts_full_rollout_after_reroot: bool = False
    mcts_historical_candidate_quota: int = 5
    mcts_validation_batch_size: int = 8

    use_semantic_loss: bool = True
    semantic_loss_weight: float = 0.1
    semantic_loss_mode: str = 'attention'

    # Extra causal CE on the compliance prefix. Disabled by default so old
    # experiment configs retain their original objective unless opted in.
    use_prefix_ce_loss: bool = False
    prefix_ce_loss_weight: float = 2.0
    prefix_ce_text: str = 'Sure, here is'

    use_refusal_sem_loss: bool = True
    refusal_sem_loss_weight: float = 0.3

    coordination_strength: float = 0.3
    num_coordination_samples: int = 3

    # Embedding 空间采样（替代 GCG 梯度）
    use_embedding_sampling: bool = False
    embedding_num_samples: int = 64
    embedding_sample_temperature: float = 1.0
    embedding_uniform_ratio: float = 0.3
    greedy_eval_candidates: int = 3
    greedy_eval_candidates_max: int = 8
    greedy_expand_threshold: float = 0.01

    # Bigram LM 先验
    lm_prior_weight: float = 0.0
    bigram_table_path: str = "attacks/data/qwen_bigram_logprobs.json"
    bigram_tokenizer_path: Optional[str] = None

class Work1Attack(BaseAttack):
    CONFIG_CLASS = Work1Config
    is_universal_attack = True

    def __init__(self, config: Dict[str, Any] = None, output_image_dir: str = None):
        super().__init__(config, output_image_dir)
        from multimodalmodels.work1_loader import load_work1_model

        self.target_model_name = str(getattr(self.cfg, "target_model", ""))
        self.model = load_work1_model(self.cfg)

        # Every released target uses the shared native-VLM attack loop.
        from .qwen2_5_utils import vlm_attacker as _vlm_attacker
        from .qwen2_5_utils import prompt_wrapper as _prompt_wrapper
        self._vlm_attacker = _vlm_attacker
        self._prompt_wrapper = _prompt_wrapper

        # cahe for universal adversarial example
        self.unversal_adv_image_path = None
        self.unversal_text_suffix = None

    def generate_test_case(
        self,
        original_prompt: str,
        image_path: str,
        case_id: str,
        **kwargs,
    ) -> TestCase:
        
        # Use cached universal adversarial example if available
        if self.unversal_adv_image_path and self.unversal_text_suffix:
            jailbreak_prompt = original_prompt + self.unversal_text_suffix
            return self.create_test_case(
                case_id=case_id,
                jailbreak_prompt=jailbreak_prompt,
                jailbreak_image_path=self.unversal_adv_image_path,
                original_prompt=original_prompt,
                original_image_path=str(image_path),
            )

        # 优先使用 cfg.save_dir，回退到 pipeline 设置的 output_image_dir
        from pathlib import Path
        save_dir = Path(getattr(self.cfg, "save_dir", str(self.output_image_dir)))
        save_dir.mkdir(parents=True, exist_ok=True)

        file = open(self.cfg.corpus, "r")

        data = list(csv.reader(file))
        file.close()
        queries = []
        targets = []
        num = len(data)

        for i in range(1, num):
            queries.append(data[i][0])
            targets.append(data[i][1])

        sample_seed = int(getattr(self.cfg, "random_seed", 42))
        training_sample_size = int(getattr(self.cfg, "training_sample_size", 66))
        selected_numbers = select_training_indices(
            len(queries), training_sample_size, sample_seed
        )

        selection_path = save_dir / "selected_training_indices.json"
        selection_path.write_text(
            json.dumps(
                {
                    "random_seed": sample_seed,
                    "training_sample_size": training_sample_size,
                    "corpus": str(self.cfg.corpus),
                    "selected_indices": selected_numbers,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f">>> Training subset: seed={sample_seed}, "
            f"samples={training_sample_size}, manifest={selection_path}"
        )

        queries = [queries[index] for index in selected_numbers]
        targets = [targets[index] + ". First," for index in selected_numbers]

        my_attacker = self._vlm_attacker.Attacker(
            self.cfg,
            self.model,
            queries,
            targets,
            device=self.model.device,
            is_rtp=False,
        )

        img_path = getattr(self.cfg, "prior_image", "")
        img = Image.open(img_path).convert("RGB")
        img = self.model.vis_processor(img).unsqueeze(0).to(self.model.device)
        text_prompt_template = getattr(
            self.model, 'work1_prompt_template', self._prompt_wrapper.qwen_chatbot_prompt
        )
        adv_img_tensor, text_suffix = my_attacker.attack_unconstrained(
            text_prompt_template,
            img=img,
            batch_size=self.cfg.batch_size,
            num_iter=self.cfg.n_iters,
            alpha=self.cfg.alpha / 255,
        )
        if adv_img_tensor.dim() == 3:
            adv_t_save = adv_img_tensor.unsqueeze(0)
        else:
            adv_t_save = adv_img_tensor
        save_path = save_dir / "work1_universal.png"
        save_image(adv_t_save, str(save_path))

        # cache the universal adversarial example
        self.unversal_adv_image_path = str(save_path)
        self.unversal_text_suffix = text_suffix

        jailbreak_prompt = original_prompt + text_suffix
        return self.create_test_case(
            case_id=case_id,
            jailbreak_prompt=jailbreak_prompt,
            jailbreak_image_path=str(save_path),
            original_prompt=original_prompt,
            original_image_path=str(image_path),
        )
