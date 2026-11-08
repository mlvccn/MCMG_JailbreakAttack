from dataclasses import asdict, is_dataclass
import re
import torch
from tqdm import tqdm
import random
from . import prompt_wrapper
from . import generator
from torchvision.utils import save_image
import gc
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

from ..minimal_gcg.opt_utils import token_gradients, sample_control
from ..minimal_gcg.opt_utils import load_model_and_tokenizer, get_filtered_cands

from ..minimal_gcg.MCS import AdvMCTSConfig, AdvMCTSSearcher
from ..minimal_gcg.MCS.base import AdvTreeNode
from ..minimal_gcg.MGC.ops import VLMOPSEnhancer
from ..run_audit import RunAudit, text_hash
from ..minimal_gcg.Semantic_similarity_loss.semantic_similarity_loss import semantic_similarity_loss


def target_prefix_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prefix_length: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Average causal CE over the first target tokens of every sample."""
    if prefix_length < 1:
        raise ValueError('prefix_length must be at least 1')
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError('Expected logits [B, T, V] and labels [B, T]')
    if logits.shape[:2] != labels.shape:
        raise ValueError('logits and labels must have matching batch/sequence dimensions')

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels.ne(ignore_index)
    target_positions = valid_mask.long().cumsum(dim=1)
    prefix_mask = valid_mask & target_positions.le(prefix_length)
    prefix_counts = prefix_mask.sum(dim=1)
    if torch.any(prefix_counts < prefix_length):
        raise ValueError(
            'At least one target is shorter than the configured prefix token sequence'
        )

    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        reduction='none',
        ignore_index=ignore_index,
    )
    sample_losses = (
        (token_losses * prefix_mask).sum(dim=1)
        / prefix_counts.to(token_losses.dtype)
    )
    return sample_losses.mean()


def snapshot_attack_config(config):
    """Return a portable snapshot of the parsed attack configuration."""
    if is_dataclass(config):
        config = asdict(config)
    elif isinstance(config, dict):
        config = dict(config)
    elif hasattr(config, '__dict__'):
        config = dict(vars(config))
    else:
        return {'value': repr(config)}

    def convert(value):
        if is_dataclass(value):
            return convert(asdict(value))
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [convert(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return repr(value)

    return convert(config)

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device)
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

class Attacker:

    def __init__(self, args, model,queries, targets, device='cuda:0', is_rtp=False):

        self.args = args
        # Freeze the launch-time values so later mutations to args or the YAML
        # file cannot change the configuration recorded in checkpoints.
        self.attack_config_snapshot = snapshot_attack_config(args)
        self.model = model
        self.device = device
        self._uses_native_adapter = bool(getattr(model, 'work1_native_backend', False))
        if self._uses_native_adapter and (
            not getattr(args, 'use_mcts', False)
            or not getattr(args, 'use_embedding_sampling', False)
        ):
            raise ValueError('Native adapters currently require MCTS + embedding sampling; '
                             'legacy embedding-only paths are unsupported')
        if self._uses_native_adapter:
            self.attack_config_snapshot['native_model'] = {
                'model_type': self.model.model_type,
                'model_path': self.model.model_path,
                'image_size': self.model.image_size,
                'patch_size': self.model.patch_size,
                'visual_tokens': self.model.image_token_estimate,
                'native_mean': list(self.model.native_mean),
                'native_std': list(self.model.native_std),
                'chat_template': self.model.work1_prompt_template,
            }
        self.is_rtp = is_rtp 
        self.queries=queries
        self.targets = targets
        self.num_targets = len(targets)

        # Native Qwen forward loss used by the image-gradient path, measured
        # after the optional suffix update and before the image update.
        self.native_loss_buffer = []
        # Backward-compatible name used by existing checkpoint analysis scripts.
        self.loss_buffer = self.native_loss_buffer

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

        # Qwen/generic interfaces used by the attacker.
        self.tokenizer = getattr(self.model, 'tokenizer', None)
        if self.tokenizer is None and hasattr(self.model, 'processor'):
            self.tokenizer = getattr(self.model.processor, 'tokenizer', None)
        if self.tokenizer is None and hasattr(self.model, 'llama_tokenizer'):
            self.tokenizer = self.model.llama_tokenizer
        if self.tokenizer is None:
            raise AttributeError('Model does not expose tokenizer or processor.tokenizer')

        self.text_model = getattr(self.model, 'language_model', None)
        if self.text_model is None:
            self.text_model = getattr(self.model, 'llama_model', self.model)

        if hasattr(self.text_model, 'model') and hasattr(self.text_model.model, 'embed_tokens'):
            self.embed_tokens = self.text_model.model.embed_tokens
        else:
            self.embed_tokens = self.text_model.get_input_embeddings()

        self.max_txt_len = getattr(self.model, 'max_txt_len', getattr(self.args, 'max_txt_len', 2048))

        # ---- Qwen2.5-VL 原生 forward 所需的 special token IDs ----
        if self._uses_native_adapter:
            self._vision_start_id = self.model.vision_start_token_id
            self._vision_end_id = self.model.vision_end_token_id
            self._image_pad_id = self.model.image_token_id
        else:
            self._vision_start_id = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            self._vision_end_id = self.tokenizer.convert_tokens_to_ids('<|vision_end|>')
            self._image_pad_id = self.tokenizer.convert_tokens_to_ids('<|image_pad|>')

        # visual token 数量: (H/patch/merge)^2, 对 448x448 图像 = 256
        H = getattr(self.model, 'image_size', 448)
        patch_size = getattr(self.model, 'patch_size', 14)
        merge_size = getattr(self.model, 'merge_size', 2)
        grid_h, grid_w = H // patch_size, H // patch_size  # 32, 32
        self._num_visual_tokens = (
            int(self.model.image_token_estimate)
            if self._uses_native_adapter
            else (grid_h // merge_size) * (grid_w // merge_size)
        )
        self._grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long, device=self.device)
        print(f" Native forward enabled: vision_start={self._vision_start_id}, "
              f"image_pad={self._image_pad_id}, visual_tokens={self._num_visual_tokens}")

        self.ops_enhancer = VLMOPSEnhancer(
            device=self.device,
            beta=0.3,
            num_sample_neighbor=2,
            num_sample_operator=3,
            sample_levels=[2],
            baseline_weight=0.6,
        )
        # 跨模态协同重加权参数
        self.coordination_strength = getattr(args, 'coordination_strength', 0.3)
        self.num_coordination_samples = getattr(args, 'num_coordination_samples', 3)
        if self.coordination_strength > 0:
            print(f" Cross-Modal Coordination enabled (strength={self.coordination_strength}, "
                  "reusing OPS gradient samples)")


        # semantic similarity loss config
        self.use_semantic_loss = getattr(args, 'use_semantic_loss', False)
        self.semantic_loss_weight = getattr(args, 'semantic_loss_weight', 0.3)
        self.semantic_loss_mode = getattr(args, 'semantic_loss_mode', 'attention')
        if self.use_semantic_loss or getattr(args, 'use_refusal_sem_loss', True):
            self.embedding_matrix = self.embed_tokens.weight.detach()
        if self.use_semantic_loss:
            print(f" Semantic Similarity Loss enabled (weight={self.semantic_loss_weight}, mode={self.semantic_loss_mode})")

        self.use_prefix_ce_loss = getattr(args, 'use_prefix_ce_loss', False)
        self.prefix_ce_loss_weight = getattr(args, 'prefix_ce_loss_weight', 2.0)
        self.prefix_ce_text = getattr(args, 'prefix_ce_text', 'Sure, here is')
        self._prefix_ce_token_ids = self.tokenizer.encode(
            self.prefix_ce_text, add_special_tokens=False
        )
        if self.use_prefix_ce_loss:
            if not self._prefix_ce_token_ids:
                raise ValueError('prefix_ce_text must tokenize to at least one token')
            mismatched_targets = [
                target for target in self.targets
                if self.tokenizer.encode(target, add_special_tokens=False)[
                    :len(self._prefix_ce_token_ids)
                ] != self._prefix_ce_token_ids
            ]
            if mismatched_targets:
                raise ValueError(
                    f'{len(mismatched_targets)} target(s) do not begin with '
                    f'prefix_ce_text={self.prefix_ce_text!r}'
                )
            print(
                f" Target Prefix CE enabled (text={self.prefix_ce_text!r}, "
                f"tokens={len(self._prefix_ce_token_ids)}, "
                f"weight={self.prefix_ce_loss_weight})"
            )

        # refusal semantic loss: 利用 semantic loss 框架，将模型输出推离拒绝语义方向
        self.use_refusal_sem_loss = getattr(args, 'use_refusal_sem_loss', True)
        self.refusal_sem_loss_weight = getattr(args, 'refusal_sem_loss_weight', 0.3)
        if self.use_refusal_sem_loss:
            self._build_refusal_embedding()
            print(f" Refusal Semantic Loss enabled (weight={self.refusal_sem_loss_weight})")

        # MCTS 对抗搜索器
        self.use_mcts = getattr(args, 'use_mcts', False)
        if self.use_mcts:
            lm_prior_weight = getattr(args, 'lm_prior_weight', 0.0)
            bigram_table_path = getattr(args, 'bigram_table_path', '')
            if self._uses_native_adapter and lm_prior_weight > 0:
                reference_path = getattr(args, 'bigram_tokenizer_path', None)
                if reference_path:
                    from transformers import AutoTokenizer
                    reference = AutoTokenizer.from_pretrained(reference_path, local_files_only=True)
                    if reference.get_vocab() != self.tokenizer.get_vocab():
                        raise ValueError('bigram_tokenizer_path does not match the selected model tokenizer')
                else:
                    print(' Bigram prior disabled: set bigram_tokenizer_path to verify '
                          'a table built for this model tokenizer')
                    lm_prior_weight, bigram_table_path = 0.0, ''
            self.attack_config_snapshot['effective_lm_prior_weight'] = lm_prior_weight
            mcts_config = AdvMCTSConfig(
                exploration_constant=getattr(args, 'exploration_constant', 0.7),
                iteration_limit=getattr(args, 'iteration_limit', 20),
                branch=getattr(args, 'branch', 16),
                topk=getattr(args, 'topk', 256),
                roll_forward_steps=getattr(args, 'roll_forward_steps', 5),
                alpha=getattr(args, 'alpha', 0.5),
                reward_beta=getattr(args, 'reward_beta', 1.0),
                backup_strategy=getattr(args, 'backup_strategy', 'mean'),
                num_candidates_return=getattr(args, 'num_candidates', 10),
                min_accept_depth=getattr(args, 'mcts_min_accept_depth', 15),
                deep_accept_improvement=getattr(
                    args, 'mcts_deep_accept_improvement', 0.02
                ),
                shallow_accept_improvement=getattr(
                    args, 'mcts_shallow_accept_improvement', 0.05
                ),
                position_stride=getattr(args, 'mcts_position_stride', 5),
                full_rollout_after_reroot=getattr(
                    args, 'mcts_full_rollout_after_reroot', False
                ),
                historical_candidate_quota=getattr(
                    args, 'mcts_historical_candidate_quota', 5
                ),
                max_branch=getattr(args, 'max_branch', 32),
                prune_keep_paths=getattr(args, 'prune_keep_paths', 10),
                verbose=getattr(args, 'verbose', True),
                # Embedding 空间采样参数（替代 GCG 梯度）
                use_embedding_sampling=getattr(args, 'use_embedding_sampling', False),
                embedding_num_samples=getattr(args, 'embedding_num_samples', 64),
                embedding_sample_temperature=getattr(args, 'embedding_sample_temperature', 1.0),
                embedding_uniform_ratio=getattr(args, 'embedding_uniform_ratio', 0.3),
                greedy_eval_candidates=getattr(args, 'greedy_eval_candidates', 3),
                greedy_eval_candidates_max=getattr(args, 'greedy_eval_candidates_max', 8),
                greedy_expand_threshold=getattr(args, 'greedy_expand_threshold', 0.01),
                # Bigram LM 先验参数
                lm_prior_weight=lm_prior_weight,
                bigram_table_path=bigram_table_path,
            )
            self.mcts_config = mcts_config
            self.mcts_searcher = AdvMCTSSearcher(
                config=mcts_config,
                model=self.model,
                device=self.device,
                suffix_loss_fn=self._mcts_native_suffix_loss,
                visible_suffix_separator='',
            )
            self.mcts_root = None
            self._mcts_suffix_tokens = None
            self._mcts_start_offset = 0
            self._mcts_visual_cache = None

            print(f" MCTS enabled (branch={mcts_config.branch}, "
                  f"rollout={mcts_config.roll_forward_steps}, "
                  f"iters={mcts_config.iteration_limit}, "
                  f"min_depth={mcts_config.min_accept_depth}, "
                  f"position_stride={mcts_config.position_stride})")

    # ================================================================
    #  Qwen2.5-VL 原生 forward 辅助方法
    #  解决: 手动构造 inputs_embeds 绕过 3D RoPE / image_mask / 多模态融合
    # ================================================================

    def _prepare_qwen_pixel_values(self, img_tensor: torch.Tensor, batch_size: int = 1):
        """将 (1, C, H, W) 图像扩展为 batch_size 份，匹配 input_ids 中的 image_pad 数量.

        Qwen2.5-VL 的 forward 要求 pixel_values 中的图像数 × 每图特征数
        等于 input_ids 中 image_pad token 总数。
        """
        return self.model.prepare_pixel_values(img_tensor, batch_size=batch_size)

    def _build_mcts_visual_cache(self, x_adv: torch.Tensor):
        """Encode one fixed image once for all suffix evaluations in an MCTS round."""
        if self._uses_native_adapter:
            return self.model.build_visual_cache(x_adv)
        pixel_values, image_grid_thw = self._prepare_qwen_pixel_values(
            x_adv, batch_size=1
        )
        with torch.no_grad():
            image_embeds = self.text_model.visual(
                pixel_values.to(dtype=self.text_model.dtype),
                grid_thw=image_grid_thw,
            )
        return {
            'image_embeds': image_embeds.detach(),
            'image_grid_thw': image_grid_thw.detach(),
        }

    @torch.no_grad()
    def _mcts_native_suffix_loss(
        self,
        x_adv: torch.Tensor,
        queries: list,
        adv_suffix: str,
        targets: list,
        text_prompt_template: str,
    ):
        return self.attack_loss_native(
            x_adv,
            queries,
            adv_suffix,
            targets,
            text_prompt_template,
            visual_cache=self._mcts_visual_cache,
        )

    def _tokenize_qwen_prompt_with_image(self, text_prompt: str, target: str):
        """Tokenize the same chat-shaped prompt used by Prompt, inserting image pads.

        Internal adv split markers are only optimization boundaries; they are not
        visible model tokens. The visible separator text before the marker is kept.
        """
        prompt_segs = re.split('<ImageHere>|<adv_split_start>|</adv_split_end>', text_prompt)
        if len(prompt_segs) < 4:
            prompt_segs = re.split('<ImageHere>', text_prompt, maxsplit=1)
            if len(prompt_segs) != 2:
                prompt_segs = ["", text_prompt]
            before_image, after_image = prompt_segs[0], prompt_segs[1]
        else:
            before_image = prompt_segs[0]
            after_image = ''.join(prompt_segs[1:])

        before_tokens = self.tokenizer.encode(
            before_image, add_special_tokens=False,
            truncation=True, max_length=self.max_txt_len,
        )
        after_tokens = self.tokenizer.encode(
            after_image, add_special_tokens=False,
            truncation=True, max_length=self.max_txt_len,
        )
        target_tokens = self.tokenizer.encode(
            target, add_special_tokens=False,
            truncation=True, max_length=self.max_txt_len,
        )

        image_tokens = (
            [self._vision_start_id]
            + [self._image_pad_id] * self._num_visual_tokens
            + [self._vision_end_id]
        )
        prefix_tokens = before_tokens + image_tokens + after_tokens
        return prefix_tokens + target_tokens, [-100] * len(prefix_tokens) + target_tokens

    def _build_refusal_embedding(self):
        """构建拒绝语义方向的 mean embedding。

        将多组常见拒绝短语的 token embedding 取均值，
        得到单个「拒绝方向」向量，用于 semantic loss 框架中的 refusal 惩罚。
        """
        refusal_phrases = [
            "I'm sorry, but I cannot",
            "I cannot provide",
            "I cannot assist",
            "It is not appropriate",
            "It is not ethical",
            "not appropriate to",
            "cannot provide guidance",
            "It is important to note",
            "never be attempted",
            "I cannot produce",
            "cannot assist with",
            "I'm unable to",
            "against the law",
            "not ethical or legal",
            "should never be attempted",
        ]
        all_token_ids = []
        for phrase in refusal_phrases:
            toks = self.tokenizer.encode(phrase, add_special_tokens=False)
            all_token_ids.extend(toks)
        # 去重
        unique_ids = list(set(all_token_ids))
        refusal_emb = self.embed_tokens.weight.detach()[torch.tensor(unique_ids, device=self.device)]
        self._refusal_embedding = refusal_emb.mean(dim=0)  # [D]
        print(f"  Refusal embedding built from {len(refusal_phrases)} phrases "
              f"({len(unique_ids)} unique tokens)")

    def _compute_refusal_sem_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                                   ignore_index: int = -100) -> torch.Tensor:
        """计算预测分布与拒绝语义的相似度。我们希望最小化该值。

        仅计算 response 位置（labels != ignore_index），
        返回 batch 内平均余弦相似度作为 loss 项。
        """
        embedding_matrix = self.embedding_matrix.to(logits.dtype)
        refusal_emb = self._refusal_embedding.to(logits.dtype)  # [D]

        # 位移: next-token prediction
        logits = logits[:, :-1, :]        # [B, T-1, V]
        labels_shifted = labels[:, 1:]    # [B, T-1]

        # 只在 response 位置计算 (labels != ignore_index)
        mask = (labels_shifted != ignore_index).float()  # [B, T-1]
        if mask.sum() < 1:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probs = torch.softmax(logits, dim=-1)               # [B, T-1, V]
        expected_emb = probs @ embedding_matrix              # [B, T-1, D]

        # 余弦相似度 vs 拒绝方向
        pred_norm = torch.nn.functional.normalize(expected_emb, dim=-1)
        ref_norm = torch.nn.functional.normalize(refusal_emb, dim=0)
        sim = (pred_norm * ref_norm).sum(dim=-1)            # [B, T-1]

        return (sim * mask).sum() / mask.sum()

    def attack_loss_native(
        self,
        x_adv: torch.Tensor,
        queries: list,
        adv_suffix: str,
        targets: list,
        text_prompt_template: str,
        visual_cache=None,
        loss_components=None,
    ):
        """通过 Qwen2.5-VL 原生 forward 计算 loss.

        关键: 将 adversarial image 作为 pixel_values 传入 model.forward(),
        让模型内部处理 3D RoPE、image_mask 和视觉-文本融合,
        梯度完整流经 visual encoder → pixel_values → adv_noise.

        Args:
            x_adv: CLIP-normalized 对抗图像 (1, C, H, W)
            queries: 当前 batch 的 query 列表
            adv_suffix: 对抗后缀字符串
            targets: 目标响应列表
            text_prompt_template: 文本模板
        Returns:
            loss (scalar tensor, retains grad graph)
        """
        if self._uses_native_adapter:
            text_batch = self.model.build_text_batch(queries, adv_suffix, targets)
            outputs = self.model.forward_native(x_adv, text_batch, visual_cache)
            return self._native_objective(outputs, text_batch['labels'], loss_components)

        batch_size = len(queries)
        separator = '<adv_split_start>'
        adv_split_end = '</adv_split_end>'

        # ---- Step 1: 准备视觉输入 ----
        if visual_cache is None:
            pixel_values, image_grid_thw = self._prepare_qwen_pixel_values(
                x_adv, batch_size=batch_size
            )
        else:
            # MCTS candidate evaluations share one fixed image. Reuse its
            # visual features and only expand the grid metadata per sample.
            pixel_values = None
            image_grid_thw = visual_cache['image_grid_thw'].expand(batch_size, -1)

        # ---- Step 2: 为每个样本构造 input_ids 和 labels ----
        all_input_ids = []
        all_labels = []

        for i in range(batch_size):
            text_prompt = text_prompt_template % (
                queries[i] + separator + adv_suffix + adv_split_end
             )
            sample_ids, sample_labels = self._tokenize_qwen_prompt_with_image(
                text_prompt, targets[i]
            )

            all_input_ids.append(torch.tensor(sample_ids, dtype=torch.long))
            all_labels.append(torch.tensor(sample_labels, dtype=torch.long))

        # ---- Step 3: Padding ----
        max_len = max(len(ids) for ids in all_input_ids)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        padded_ids, padded_labels, attn_mask = [], [], []
        for i in range(batch_size):
            seq_len = len(all_input_ids[i])
            pad_len = max_len - seq_len
            padded_ids.append(torch.cat([
                all_input_ids[i],
                torch.full((pad_len,), pad_token_id, dtype=torch.long),
            ]).unsqueeze(0))
            padded_labels.append(torch.cat([
                all_labels[i],
                torch.full((pad_len,), -100, dtype=torch.long),
            ]).unsqueeze(0))
            attn_mask.append(torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]).unsqueeze(0))

        input_ids = torch.cat(padded_ids, dim=0).to(self.device)
        labels = torch.cat(padded_labels, dim=0).to(self.device)
        attention_mask = torch.cat(attn_mask, dim=0).to(self.device)

        # ---- Step 4: 原生 forward ----
        if visual_cache is None:
            # Image gradients flow through the visual encoder in this path.
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(dtype=self.text_model.dtype),
                image_grid_thw=image_grid_thw,
                labels=labels,
                use_cache=False,
            )
        else:
            # Match Qwen's internal image-token replacement exactly, but use
            # the already-computed visual features. Passing input_ids keeps
            # the model's native 3D RoPE index calculation enabled.
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
            image_embeds = visual_cache['image_embeds'].to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            image_embeds = image_embeds.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            image_mask = (input_ids == self._image_pad_id).unsqueeze(-1)
            image_mask = image_mask.expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds
            )
            outputs = self.text_model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                labels=labels,
                use_cache=False,
            )
        return self._native_objective(outputs, labels, loss_components)

    def _native_objective(self, outputs, labels, loss_components=None):
        """Use one objective for native image optimization and suffix evaluation."""
        ce_loss = outputs.loss
        prefix_ce_loss = None
        sem_loss = None
        refusal_loss = None

        # ---- Step 5: 强调决定自由生成首个分支的服从前缀 ----
        loss = ce_loss
        if self.use_prefix_ce_loss:
            prefix_ce_loss = target_prefix_cross_entropy(
                outputs.logits,
                labels,
                prefix_length=len(self._prefix_ce_token_ids),
                ignore_index=-100,
            )
            loss = loss + self.prefix_ce_loss_weight * prefix_ce_loss

        # ---- Step 6: 可选的 semantic similarity loss ----
        if self.use_semantic_loss:
            sem_loss = semantic_similarity_loss(
                outputs.logits, labels,
                self.embedding_matrix.to(outputs.logits.dtype),
                mode=self.semantic_loss_mode, ignore_index=-100,
            )
            loss = loss + self.semantic_loss_weight * sem_loss

        # ---- Step 7: 拒绝语义惩罚 (推离 refusal 方向) ----
        if self.use_refusal_sem_loss:
            refusal_loss = self._compute_refusal_sem_loss(
                outputs.logits, labels, ignore_index=-100
            )
            loss = loss + self.refusal_sem_loss_weight * refusal_loss

        if loss_components is not None:
            loss_components.update({
                'ce_loss': ce_loss.detach(),
                'prefix_ce_loss': prefix_ce_loss.detach() if prefix_ce_loss is not None else ce_loss.detach().new_zeros(()),
                'semantic_loss': sem_loss.detach() if sem_loss is not None else ce_loss.detach().new_zeros(()),
                'refusal_loss': refusal_loss.detach() if refusal_loss is not None else ce_loss.detach().new_zeros(()),
                'total_loss': loss.detach(),
            })
        return loss

    # ================================================================
    #  主攻击循环
    # ================================================================

    def attack_unconstrained(self, text_prompt, img, adv_string_init='! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !',
                             batch_size = 8, num_iter=2000, alpha=1/255):

        print('>>> batch_size:', batch_size)

        my_generator = None if self._uses_native_adapter else generator.Generator(model=self.model)


        adv_noise = denormalize(img).clone().to(self.device)
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        eps_pixel = float(getattr(self.args, 'eps', 32)) / 255.0
        if eps_pixel < 0:
            raise ValueError(f'eps must be non-negative, got {self.args.eps}')
        _original_image = adv_noise.detach().clone()
        print(f'>>> L-inf budget: eps={eps_pixel:.6f} ({eps_pixel * 255:.1f}/255)')

        adv_suffix = adv_string_init

        # 新增参数，初期噪声大小
        initial_noise_std = 0.02

        # P0 改进: 为 MCTS 采样固定的评估 batch，消除跨搜索轮次的 batch 不一致
        # 使用与主循环相同的 batch_size=8，避免显存翻倍
        mcts_batch_size = min(8, len(self.queries))
        requested_validation_size = max(
            0, int(getattr(self.args, 'mcts_validation_batch_size', 8))
        )
        validation_batch_size = min(
            requested_validation_size,
            len(self.queries) - mcts_batch_size,
        )
        # 搜索/验证划分对比实验必须一致，MCTS 内部探索仍使用全局 RNG。
        mcts_batch_rng = random.Random(
            int(getattr(self.args, 'random_seed', 42)) + 1009
        )
        sampled_eval_indices = mcts_batch_rng.sample(
            range(len(self.queries)),
            mcts_batch_size + validation_batch_size,
        )
        mcts_sample_indices = sampled_eval_indices[:mcts_batch_size]
        mcts_validation_indices = sampled_eval_indices[mcts_batch_size:]
        mcts_queries = [self.queries[i] for i in mcts_sample_indices]
        mcts_targets = [self.targets[i] for i in mcts_sample_indices]
        mcts_validation_queries = [
            self.queries[i] for i in mcts_validation_indices
        ]
        mcts_validation_targets = [
            self.targets[i] for i in mcts_validation_indices
        ]
        audit = RunAudit(
            self.args.save_dir, self.attack_config_snapshot,
            mcts_sample_indices, self.queries,
            validation_indices=mcts_validation_indices,
        )
        print(f'>>> MCTS fixed eval batch: {mcts_batch_size} search + '
              f'{validation_batch_size} validation samples')

        text_update_interval = 10
        print(f'>>> Text update interval: {text_update_interval}')

        # The legacy GCG branch compares suffixes on a noisy image. MCTS
        # evaluates directly on x_adv and does not need this extra encoding.
        use_robust_text_prompt = not self.use_mcts
        needs_legacy_prompt = use_robust_text_prompt
        needs_legacy_loss = not self.use_mcts

        for t in tqdm(range(num_iter + 1)):

            batch_queries_targets = random.sample(list(zip(self.queries,self.targets)), batch_size)
            batch_queries=[t[0] for t in batch_queries_targets]
            batch_targets = [t[1] for t in batch_queries_targets]
            adv_split_start='<adv_split_start>'
            separator = adv_split_start
            adv_split_end = '</adv_split_end>'
            text_prompts = [text_prompt % (batch_queries[i]+separator+adv_suffix+adv_split_end) for i in range(batch_size)]


            x_adv = normalize(adv_noise)

            prompt = None
            if needs_legacy_prompt:
                prompt = prompt_wrapper.Prompt(
                    model=self.model,
                    text_prompts=text_prompts,
                    img_prompts=[[x_adv]],
                )
                prompt.img_embs = prompt.img_embs * batch_size
                prompt.update_context_embs()

            # Only the legacy GCG token-gradient branch requires this manual
            # embedding loss. MCTS suffix evaluation and image gradients
            # both use attack_loss_native.
            if needs_legacy_loss:
                target_loss = self.attack_loss(prompt, batch_targets)
                target_loss.backward()

            if t % text_update_interval == 0:
                if use_robust_text_prompt:
                    # The noisy view is retained only for text optimizers whose
                    # candidate and baseline losses both use that same view.
                    current_std = initial_noise_std * (1 - t / num_iter)
                    noise = torch.randn_like(adv_noise) * current_std
                    adv_noise_robust = (adv_noise.detach() + noise).clamp(0, 1)
                    adv_noise_robust.requires_grad_(True)
                    x_adv_robust = normalize(adv_noise_robust)
                    prompt_robust = prompt_wrapper.Prompt(
                        model=self.model,
                        text_prompts=text_prompts,
                        img_prompts=[[x_adv_robust]],
                    )
                    prompt_robust.img_embs = prompt_robust.img_embs * batch_size
                    prompt_robust.update_context_embs()

                if self.use_mcts:
                    # ===================== MCTS 搜索分支 (P0 改进: 固定 batch) =====================
                    gc.collect()
                    torch.cuda.empty_cache()
                    # All suffix evaluations below share this image and fixed
                    # MCTS batch. Cache visual features for this round only.
                    self.mcts_searcher.begin_evaluation_round()
                    suffix_hash_before = text_hash(adv_suffix)
                    search_records = []
                    search_baseline_loss = None
                    validation_baseline_loss = None
                    validation_candidate_loss = None
                    validation_improvement = None
                    mean_improvement = None
                    required_improvement = None
                    search_accepted = False
                    search_reason = 'no_candidates'
                    self._mcts_visual_cache = self._build_mcts_visual_cache(x_adv)

                    if hasattr(self, '_mcts_suffix_tokens') and self._mcts_suffix_tokens is not None:
                        adv_suffix_tokens_init = self._mcts_suffix_tokens.to(self.device)
                    else:
                        adv_suffix_tokens_init = self.tokenizer.encode(
                            adv_suffix, add_special_tokens=False, return_tensors='pt'
                        ).squeeze(0).to(self.device)
                    # P0 改进: 使用固定的 MCTS 评估 batch，确保树节点 V 值跨轮次可比
                    _mcts_batch_queries = mcts_queries
                    _mcts_batch_targets = mcts_targets
                    _mcts_batch_size = mcts_batch_size
                    new_suffix = self.tokenizer.decode(adv_suffix_tokens_init, skip_special_tokens=True)
                    if self.mcts_root is None \
                       or not hasattr(self, '_mcts_root_suffix') \
                       or not torch.equal(adv_suffix_tokens_init, self._mcts_root_suffix):
                        # 首次创建 / 后缀发生变化 → 重建根节点
                        if self.mcts_root is not None:
                            old = self.tokenizer.decode(self._mcts_root_suffix, skip_special_tokens=True)
                            print(f"[MCTS] Re-rooting tree: '{old[:50]}' -> '{new_suffix[:50]}'")
                        self._mcts_root_suffix = adv_suffix_tokens_init.clone()
                        self.mcts_root = AdvTreeNode(
                            token_id=-1,
                            parent=None,
                            depth=0,
                            base_suffix_tokens=adv_suffix_tokens_init,
                            start_offset=self._mcts_start_offset,
                        )
                        root_loss, root_reward = self.mcts_searcher.evaluate_suffix(
                            adv_suffix_tokens_init, x_adv, _mcts_batch_queries, _mcts_batch_targets,
                            text_prompt, _mcts_batch_size, prompt_wrapper.Prompt, self.attack_loss
                        )
                        self.mcts_root.update_value(
                            root_reward,
                            root_loss,
                            iteration=t,
                            evaluation_round=self.mcts_searcher.evaluation_round,
                        )
                        self.mcts_root.numVisits = 1
                        print(f"[MCTS] Root loss={root_loss:.4f}, "
                              f"start_offset={self.mcts_root.start_offset}, "
                              f"suffix='{new_suffix[:50]}'")
                    else:
                        # 后缀未变，仅图像更新 → 衰减旧值
                        old_root_V = self.mcts_root.V
                        _, new_root_reward = self.mcts_searcher.evaluate_suffix(
                            adv_suffix_tokens_init, x_adv, _mcts_batch_queries, _mcts_batch_targets,
                            text_prompt, _mcts_batch_size, prompt_wrapper.Prompt, self.attack_loss
                        )
                        if old_root_V > 1e-10:
                            scale = new_root_reward / old_root_V
                        else:
                            scale = 1.0
                        self.mcts_root.rescale_tree_values(scale, pw_alpha=self.mcts_config.pw_alpha)
                        self.mcts_root.V = new_root_reward
                    # MCTS 搜索返回多个候选后缀（内部所有评估均使用固定 batch）
                    candidates = self.mcts_searcher.search_continue(
                        root=self.mcts_root,
                        x_adv=x_adv,
                        batch_queries=_mcts_batch_queries,
                        batch_targets=_mcts_batch_targets,
                        text_prompt=text_prompt,
                        batch_size=_mcts_batch_size,
                        prompt_wrapper_cls=prompt_wrapper.Prompt,
                        attack_loss_fn=self.attack_loss,
                    )
                    mcts_rerooted = False

                    # 在当前批次上重新评估所有候选（使用固定 batch 确保可比性）
                    if candidates:
                        evaluated_candidates = self.mcts_searcher.evaluate_candidates_batch(
                            candidates, x_adv, _mcts_batch_queries, _mcts_batch_targets,
                            text_prompt, _mcts_batch_size, prompt_wrapper.Prompt, self.attack_loss
                        )

                        # Baseline 和候选都通过同一 token 解码及 Qwen
                        # native forward 路径评估，保证 loss 可直接比较。
                        clean_baseline_loss, _ = self.mcts_searcher.evaluate_suffix(
                            adv_suffix_tokens_init,
                            x_adv,
                            _mcts_batch_queries,
                            _mcts_batch_targets,
                            text_prompt,
                            _mcts_batch_size,
                            prompt_wrapper.Prompt,
                            self.attack_loss,
                        )
                        search_baseline_loss = clean_baseline_loss

                        # 深度门控：优先选择足够深的候选；浅层候选只有在
                        # loss 改善显著时才能触发 re-root。
                        suffix_len = self.mcts_root.suffix_len
                        min_accept_depth = (
                            self.mcts_config.normalized_min_accept_depth(suffix_len)
                        )
                        admissible_candidates = []
                        for cand, loss in evaluated_candidates:
                            improvement = clean_baseline_loss - loss
                            candidate_required_improvement = (
                                self.mcts_config.required_accept_improvement(
                                    cand.depth, suffix_len
                                )
                            )
                            candidate_is_eligible = self.mcts_config.should_accept_candidate(
                                cand.depth, improvement, suffix_len
                            )
                            if candidate_is_eligible:
                                admissible_candidates.append((cand, loss))
                            search_records.append({
                                'suffix_sha256': text_hash(cand.suffix_str),
                                'recall_source': cand.recall_source,
                                'depth': cand.depth,
                                'loss': loss,
                                'improvement': improvement,
                                'required_improvement': candidate_required_improvement,
                                'eligible': candidate_is_eligible,
                                'reason': ('no_improvement' if improvement <= 0 else
                                           'eligible' if candidate_is_eligible else
                                           'below_accept_threshold'),
                            })

                        best_overall_cand, best_overall_loss = evaluated_candidates[0]

                        print(f"[MCTS] Baseline loss: {clean_baseline_loss:.4f} "
                              f"(n={_mcts_batch_size})")
                        print(f"[MCTS] Best candidate loss: {best_overall_loss:.4f} "
                              f"(depth={best_overall_cand.depth}, "
                              f"min_accept_depth={min_accept_depth})")

                        if admissible_candidates and validation_batch_size > 0:
                            proposal_cand, proposal_loss = admissible_candidates[0]
                            validation_baseline_loss = self._mcts_native_suffix_loss(
                                x_adv,
                                mcts_validation_queries,
                                self.tokenizer.decode(
                                    adv_suffix_tokens_init,
                                    skip_special_tokens=True,
                                ),
                                mcts_validation_targets,
                                text_prompt,
                            ).item()
                            validation_candidate_loss = self._mcts_native_suffix_loss(
                                x_adv,
                                mcts_validation_queries,
                                proposal_cand.suffix_str,
                                mcts_validation_targets,
                                text_prompt,
                            ).item()
                            proposal_improvement = clean_baseline_loss - proposal_loss
                            validation_improvement = (
                                validation_baseline_loss - validation_candidate_loss
                            )
                            mean_improvement = (
                                proposal_improvement + validation_improvement
                            ) / 2.0
                            required_improvement = (
                                self.mcts_config.required_accept_improvement(
                                    proposal_cand.depth, suffix_len
                                )
                            )
                            print(
                                f"[MCTS] Validation: "
                                f"{validation_baseline_loss:.4f} -> "
                                f"{validation_candidate_loss:.4f}, "
                                f"improvement={validation_improvement:.4f}, "
                                f"mean={mean_improvement:.4f}, "
                                f"required={required_improvement:.4f}"
                            )
                            validation_passed = self.mcts_config.passes_validation(
                                proposal_cand.depth,
                                proposal_improvement,
                                validation_improvement,
                                suffix_len,
                            )
                            if not validation_passed:
                                admissible_candidates = []
                                search_reason = (
                                    'validation_no_improvement'
                                    if validation_improvement <= 0 else
                                    'validation_below_threshold'
                                )

                        if admissible_candidates:
                            search_accepted = True
                            search_reason = 'accepted'
                            best_cand, best_loss = admissible_candidates[0]
                            adv_suffix = best_cand.suffix_str
                            accepted_tokens = best_cand.suffix_tokens.detach().clone().to(self.device)
                            previous_tokens = self._mcts_suffix_tokens
                            if previous_tokens is None:
                                previous_tokens = self.mcts_root._base_suffix_tokens
                            suffix_changed = not torch.equal(
                                accepted_tokens, previous_tokens.to(self.device)
                            )
                            self._mcts_suffix_tokens = accepted_tokens
                            self._mcts_root_suffix = accepted_tokens.clone()

                            # 候选被接受后，以完整新后缀为 baseline 重新建树。
                            # 旧树的路径、统计量和候选池均依赖旧 baseline，不能继续复用。
                            if suffix_changed:
                                previous_offset = self.mcts_root.start_offset
                                self._mcts_start_offset = self.mcts_config.next_start_offset(
                                    previous_offset, suffix_len
                                )
                                root_reward = self.mcts_config.loss_to_reward(best_loss)
                                self.mcts_root = AdvTreeNode(
                                    token_id=-1,
                                    parent=None,
                                    depth=0,
                                    base_suffix_tokens=accepted_tokens,
                                    start_offset=self._mcts_start_offset,
                                )
                                self.mcts_root.update_value(
                                    root_reward,
                                    best_loss,
                                    iteration=t,
                                    evaluation_round=self.mcts_searcher.evaluation_round,
                                )
                                self.mcts_root.numVisits = 1
                                mcts_rerooted = True
                                if self.mcts_config.full_rollout_after_reroot:
                                    rollout_node = self.mcts_root
                                    for _ in range(
                                        0, suffix_len,
                                        self.mcts_config.roll_forward_steps,
                                    ):
                                        _, rollout_node = self.mcts_searcher.greedy_rollout(
                                            rollout_node, x_adv,
                                            _mcts_batch_queries, _mcts_batch_targets,
                                            text_prompt, _mcts_batch_size,
                                            prompt_wrapper.Prompt, self.attack_loss,
                                            current_iteration=t,
                                        )
                                        if (rollout_node is None or
                                                rollout_node.depth >= suffix_len):
                                            break
                                    self.mcts_root.numVisits = 1
                                    rollout_depth = (
                                        rollout_node.depth
                                        if rollout_node is not None else 0
                                    )
                                    print(f"[MCTS] Full rollout to depth {rollout_depth}")
                            print(f"[MCTS] Updated adv_suffix (loss: {clean_baseline_loss:.4f} -> {best_loss:.4f})")
                            print(f"[MCTS] New suffix: {adv_suffix}")
                            if mcts_rerooted:
                                print(f"[MCTS] Re-rooted tree on accepted suffix "
                                      f"(start_offset={self.mcts_root.start_offset})")
                            if len(evaluated_candidates) > 1:
                                print(f"[MCTS] Top candidates (fixed batch):")
                                for i, (cand, loss) in enumerate(evaluated_candidates[:3]):
                                    print(f"[MCTS]   #{i+1}: loss={loss:.4f}, depth={cand.depth}")
                        elif search_reason.startswith('validation_'):
                            print("[MCTS] Candidate rejected by validation batch.")
                        elif best_overall_loss < clean_baseline_loss:
                            improvement = clean_baseline_loss - best_overall_loss
                            required = self.mcts_config.required_accept_improvement(
                                best_overall_cand.depth, suffix_len
                            )
                            search_reason = 'no_eligible_candidate'
                            print(f"[MCTS] Improvement below acceptance threshold: "
                                  f"depth={best_overall_cand.depth}, "
                                  f"improvement={improvement:.4f} < {required:.4f}")
                        else:
                            search_reason = 'no_eligible_candidate'
                            print(f"[MCTS] No improvement, keeping current suffix.")

                    audit.append('search', {
                        'iteration': t,
                        'evaluation_round': self.mcts_searcher.evaluation_round,
                        'baseline_loss': search_baseline_loss,
                        'validation_baseline_loss': validation_baseline_loss,
                        'validation_candidate_loss': validation_candidate_loss,
                        'validation_improvement': validation_improvement,
                        'mean_improvement': mean_improvement,
                        'required_improvement': required_improvement,
                        'returned_candidates': len(candidates),
                        'recall_source_counts': {
                            source: sum(c.recall_source == source for c in candidates)
                            for source in sorted({c.recall_source for c in candidates})
                        },
                        'depth_counts': {
                            str(depth): sum(c.depth == depth for c in candidates)
                            for depth in sorted({c.depth for c in candidates})
                        },
                        'candidates': search_records,
                        'unique_evaluated_suffixes': len(self.mcts_searcher._suffix_loss_cache or {}),
                        'accepted': search_accepted,
                        'suffix_changed': text_hash(adv_suffix) != suffix_hash_before,
                        'suffix_sha256_before': suffix_hash_before,
                        'suffix_sha256_after': text_hash(adv_suffix),
                        'reason': search_reason,
                    })
                    # 修剪树：只保留最优路径，防止无限膨胀
                    if (candidates and not mcts_rerooted and
                            self.mcts_config.prune_keep_paths > 0):
                        # 用评估后的候选节点（已按 loss 排序）
                        if evaluated_candidates:
                            keep_nodes = [c.node for c, _ in evaluated_candidates[:self.mcts_config.prune_keep_paths] if c.node is not None]
                        else:
                            keep_nodes = [c.node for c in candidates[:self.mcts_config.prune_keep_paths] if c.node is not None]
                        kept = self.mcts_root.prune_to_paths(keep_nodes)
                        print(f"[MCTS] Pruned tree to {kept} nodes (kept {len(keep_nodes)} paths)")

                    stats = self.mcts_root.get_tree_stats()
                    print(f"[MCTS] Tree total: nodes={stats['total_nodes']}, "
                          f"max_depth={stats['max_depth']}, min_loss={stats['min_loss']:.4f}")

                    self._mcts_visual_cache = None
                    gc.collect()
                    torch.cuda.empty_cache()

                else:
                    text_grad=None
                    for i in range(batch_size):#Aggregate gradients
        
                        new_grad=token_gradients(self.model, prompt.adv_suffix_tokens, prompt.adv_len,prompt.offset[i],
                                                prompt.context_embs[i].grad)
                        if text_grad is None:
                            text_grad=torch.zeros_like(new_grad)
                        text_grad+=new_grad
                    with torch.no_grad():
                        # 计算加噪声后损失
                        robust_loss = self.attack_loss(prompt_robust, batch_targets)

                        new_adv_suffix_toks=sample_control(prompt.adv_suffix_tokens,text_grad,
                                                    batch_size=250, topk=256)

                        candidates = get_filtered_cands(self.tokenizer,
                                                            new_adv_suffix_toks,
                                                            filter_cand=True,
                                                            curr_control=adv_suffix)

                        curr_best_loss = 9999
                        curr_best_trigger = None
         
                        for cand in candidates:
                            cand_text_prompts = [text_prompt % (batch_queries[i] + separator + cand + adv_split_end) for i in
                                            range(batch_size)]
                            # cand_prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=cand_text_prompts, img_prompts=[[x_adv]])
                            # 使用 x_adv_robust
                            cand_prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=cand_text_prompts, img_prompts=[[x_adv_robust]])
                            cand_prompt.img_embs = cand_prompt.img_embs * batch_size
                            cand_prompt.update_cand_context_embs()
                            next_target_loss = self.attack_loss(cand_prompt, batch_targets)
                            curr_loss = next_target_loss

                            if curr_loss < curr_best_loss:
                                curr_best_loss = curr_loss
                                curr_best_trigger = cand
           
                    print("curr_best_loss",curr_best_loss)
                    print("target_loss",target_loss)
                    # if curr_best_loss < target_loss:
        
                    #     adv_suffix = curr_best_trigger

                    if curr_best_loss < robust_loss.item():
                            adv_suffix = curr_best_trigger
                            print("Updated adv_suffix:", adv_suffix)
                        
                    del prompt_robust, robust_loss, text_grad,new_adv_suffix_toks ; gc.collect() 
                # del text_grad ; gc.collect() 
            native_baseline_loss = None
            native_components = {}

            def compute_grad_fn(img_01, transform=None, *, grad_queries=None,
                                grad_targets=None, grad_suffix=None,
                                record_baseline=True):
                nonlocal native_baseline_loss
                grad_queries = batch_queries if grad_queries is None else grad_queries
                grad_targets = batch_targets if grad_targets is None else grad_targets
                grad_suffix = adv_suffix if grad_suffix is None else grad_suffix
                img_01 = img_01.detach().clone().requires_grad_(True)
                img = transform(img_01) if transform is not None else img_01
                x_adv_norm = normalize(img)
                # 原生 forward, sub_batch=2 避免 OOM
                # 按子 batch 大小加权: d(loss)/dx = Σ (n_i/N) * d(loss_i)/dx
                sub_batch = int(getattr(self.args, 'image_gradient_sub_batch_size', 2))
                if sub_batch < 1:
                    raise ValueError('image_gradient_sub_batch_size must be positive')
                total_n = len(grad_queries)
                grad_accum = None
                loss_accum = 0.0
                for _s in range(0, total_n, sub_batch):
                    sub_q = grad_queries[_s:_s + sub_batch]
                    sub_t = grad_targets[_s:_s + sub_batch]
                    n_i = len(sub_q)
                    retain = (_s + sub_batch < total_n)
                    components = {} if transform is None and record_baseline else None
                    raw_loss = self.attack_loss_native(
                        x_adv_norm, sub_q, grad_suffix, sub_t, text_prompt,
                        loss_components=components,
                    )
                    batch_weight = n_i / total_n
                    sub_loss = raw_loss * batch_weight  # 按子 batch 占比加权
                    loss_accum += raw_loss.detach().float().item() * batch_weight
                    if components is not None:
                        for name, value in components.items():
                            native_components[name] = native_components.get(name, 0.0) + value.float().item() * batch_weight
                    sub_grad = torch.autograd.grad(sub_loss, img_01, retain_graph=retain)[0]
                    if grad_accum is None:
                        grad_accum = sub_grad.detach().clone()
                    else:
                        grad_accum = grad_accum + sub_grad.detach()
                    del raw_loss, sub_loss, sub_grad
                # OPS always evaluates this untransformed baseline once. Reuse
                # that forward to record the native objective at no extra cost.
                if transform is None and record_baseline:
                    native_baseline_loss = loss_accum
                gc.collect()
                torch.cuda.empty_cache()
                return grad_accum

            avg_grad, stability_soft_mask = (
                self.ops_enhancer.get_averaged_gradient_with_coordination(
                    adv_noise,
                    compute_grad_fn,
                    coordination_strength=self.coordination_strength,
                    num_coordination_samples=self.num_coordination_samples,
                    return_soft_mask=True,
                )
            )
            if native_baseline_loss is None:
                raise RuntimeError('OPS did not evaluate the native baseline loss')

            with torch.no_grad():
                # Apply the continuous stability mask after sign(), so it controls
                # the per-pixel step size instead of being cancelled by sign().
                updated_image = adv_noise - alpha * stability_soft_mask * avg_grad.sign()
                updated_image = torch.max(
                    torch.min(updated_image, _original_image + eps_pixel),
                    _original_image - eps_pixel,
                )
                adv_noise.copy_(updated_image.clamp(0, 1))
            if adv_noise.grad is not None:
                adv_noise.grad.zero_()
            self.model.zero_grad()

            self.native_loss_buffer.append(native_baseline_loss)
            audit.append('loss_components', {
                'iteration': t,
                'measurement': 'untransformed_baseline_after_text_before_image_update',
                'query_sha256': [text_hash(query) for query in batch_queries],
                'suffix_sha256': text_hash(adv_suffix),
                'weighted_prefix_ce_loss': self.prefix_ce_loss_weight * native_components.get('prefix_ce_loss', 0.0),
                'weighted_semantic_loss': self.semantic_loss_weight * native_components.get('semantic_loss', 0.0),
                'weighted_refusal_loss': self.refusal_sem_loss_weight * native_components.get('refusal_loss', 0.0),
                **native_components,
            })

            print("native_image_loss: %.6f" % native_baseline_loss)

            self.model.zero_grad()

            if t % 20 == 0:
                self.plot_loss()

            if t % 50 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = normalize(adv_noise)
                if self._uses_native_adapter:
                    response = self.model.generate_native(
                        x_adv, batch_queries[0] + adv_suffix,
                        max_new_tokens=getattr(self.args, 'preview_max_new_tokens', 300),
                    )
                else:
                    preview_text_prompts = [
                        text_prompt % (
                            batch_queries[i] + separator + adv_suffix + adv_split_end
                        )
                        for i in range(batch_size)
                    ]
                    preview_prompt = prompt_wrapper.Prompt(
                        model=self.model,
                        text_prompts=preview_text_prompts,
                        img_prompts=[[x_adv]],
                    )
                    preview_prompt.img_embs = preview_prompt.img_embs * batch_size
                    preview_prompt.update_context_embs()
                    with torch.no_grad():
                        response, _ = my_generator.generate(preview_prompt)
                    del preview_prompt
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                # save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))
                print("adv_suffix:",adv_suffix)

                # with open("adv_suffix.txt", "a") as file:

                #     file.write(adv_suffix + "\n")

            if t % 500 == 0 and t > 0:
                save_dir = getattr(self.args, 'save_dir', 'output/test_cases/work1')
                # 保存 PNG 格式的对抗图像（便于直接查看）
                save_image(adv_img_prompt, f'{save_dir}/work1_universal_iter_{t}.png')
                # 保存完整 checkpoint（含图像 tensor、后缀、loss）
                checkpoint = {
                    'iteration': t,
                    'adv_suffix': adv_suffix,
                    'adv_img_prompt': adv_img_prompt,
                    'loss_buffer': self.loss_buffer,
                    'native_loss_buffer': self.native_loss_buffer,
                    'eps_pixel': eps_pixel,
                    'attack_config': self.attack_config_snapshot,
                    'audit_run_id': audit.run_id,
                    'mcts_indices_in_training_subset': mcts_sample_indices,
                    'mcts_validation_indices_in_training_subset': mcts_validation_indices,
                }
                torch.save(checkpoint, f'{save_dir}/checkpoint_iter_{t}.pt')
                audit.save_rng(f'iter_{t}')
                print(f"Checkpoint saved at iteration {t}")
 
        adv_img_prompt = adv_noise.detach().cpu().squeeze(0)
        return adv_img_prompt, adv_suffix



    def plot_loss(self):

        sns.set_theme()
        num_iters = len(self.native_loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.native_loss_buffer, label='Native Pre-Image Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()

        torch.save(self.native_loss_buffer, '%s/loss' % (self.args.save_dir))
        torch.save(
            self.native_loss_buffer,
            '%s/native_loss' % (self.args.save_dir),
        )

    def attack_loss(self, prompts, targets):

        context_embs = prompts.context_embs

        if len(context_embs) == 1:
            context_embs = context_embs * len(targets) # expand to fit the batch_size

        assert len(context_embs) == len(targets), f"Unmathced batch size of prompts and targets {len(context_embs)} != {len(targets)}"

        batch_size = len(targets)
        self.tokenizer.padding_side = "right"
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
 
  
        to_regress_tokens = self.tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(self.device)
        to_regress_embs = self.embed_tokens(to_regress_tokens.input_ids)

        pad = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * pad_token_id
        pad_embs = self.embed_tokens(pad)


        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == pad_token_id, -100
        )


        pos_padding = torch.argmin(T, dim=1) # a simple trick to find the start position of padding

        input_embs = []
        targets_mask = []

        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []

        for i in range(batch_size):

            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]

            targets_mask.append(T[i:i+1, :target_length])
            input_embs.append(to_regress_embs[i:i+1, :target_length]) # omit the padding tokens

            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length

            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)

        max_length = max(seq_tokens_length)

        attention_mask = []

        for i in range(batch_size):

            # masked out the context from loss computation
            context_mask =(
                torch.ones([1, context_tokens_length[i]],
                       dtype=torch.long).to(self.device).fill_(-100)
            )

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = (
                torch.ones([1, num_to_pad],
                       dtype=torch.long).to(self.device).fill_(-100)
            )

            targets_mask[i] = torch.cat( [context_mask, targets_mask[i], padding_mask], dim=1 )
            input_embs[i] = torch.cat( [context_embs[i], input_embs[i],
                                        pad_embs.repeat(1, num_to_pad, 1)], dim=1 )
            attention_mask.append( torch.LongTensor( [[1]* seq_tokens_length[i] + [0]*num_to_pad ] ) )

        targets = torch.cat( targets_mask, dim=0 ).to(self.device)
        inputs_embs = torch.cat( input_embs, dim=0 ).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)


        outputs = self.text_model(
                inputs_embeds=inputs_embs,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        # Semantic similarity loss: use attention-based semantic distance
        if self.use_semantic_loss:
            logits = outputs.logits  # [B, seq_len, V]
            sem_loss = semantic_similarity_loss(
                logits, targets, self.embedding_matrix.to(logits.dtype),
                mode=self.semantic_loss_mode, ignore_index=-100
            )
            loss = loss + self.semantic_loss_weight * sem_loss

        # Refusal semantic penalty: push model away from refusal direction
        if self.use_refusal_sem_loss:
            refusal_loss = self._compute_refusal_sem_loss(
                outputs.logits, targets, ignore_index=-100
            )
            loss = loss - self.refusal_sem_loss_weight * refusal_loss

        return loss
