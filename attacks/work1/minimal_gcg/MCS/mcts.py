"""
基于过程奖励模型的 MCTS 对抗后缀搜索器 —— 逐位置搜索版本。

搜索树结构：
- 深度 0 (根节点): 初始对抗后缀，不代表任何 token 选择
- 深度 d (1 ≤ d ≤ suffix_len): 循环顺序中第 d 个位置的 token 选择
- 实际位置 = (start_offset + d - 1) % suffix_len
- 从根到深度 d 的路径 = 循环顺序中前 d 个位置的确定性 token 替换
- 剩余位置保持初始后缀的原始值

重要说明：
- 每次 MCTS 搜索使用不同的 batch，节点的 V/loss 仅反映该节点被评估时的批次性能
- 因此搜索结束后，返回多个候选后缀，由调用方在当前批次上重新评估选择最优

每次 MCTS 迭代：
1. Selection:  UCB 向下遍历至未扩展节点
2. Expansion:  Embedding空间采样 Top-k token 作为候选子节点
3. Rollout:    从该位置起，逐位置采样+评估选择最优 token
4. Backprop:   将最优奖励沿路径反向传播

Embedding 空间采样策略 (v2):
- 替代 GCG 梯度，从 embedding 空间采样候选 token
- 混合策略: embedding 相似度采样 (70%) + 均匀随机采样 (30%)
- 在 greedy_rollout 中评估多个候选，按实际 loss 选择最优
- 无需 backward pass，消除"假更新"问题
"""

import torch
import math
import random
import gc
from typing import Callable, List, Tuple, Optional

import torch.nn.functional as F
from ..opt_utils import token_gradients
from .base import AdvTreeNode, AdvMCTSConfig, SuffixCandidate


class AdvMCTSSearcher:
    """对抗后缀逐位置 MCTS 搜索器"""

    def __init__(self, config: AdvMCTSConfig, model, device='cuda:0',
                 suffix_loss_fn: Optional[Callable] = None,
                 visible_suffix_separator: str = ' | '):
        """
        Args:
            config: MCTS 配置
            model: 目标 VLM 模型（用于 tokenizer 和梯度计算）
            device: 计算设备
            suffix_loss_fn: 可选的原生后缀评估函数，签名为
                (x_adv, queries, suffix, targets, text_prompt) -> scalar loss
            visible_suffix_separator: query 和后缀之间对模型可见的字符串
        """
        self.config = config
        self.model = model
        self.device = device
        self.suffix_loss_fn = suffix_loss_fn
        self.visible_suffix_separator = visible_suffix_separator
        # Enabled only for one fixed (image, batch) MCTS round. The caller
        # must start a new round after the image changes.
        self._suffix_loss_cache = None
        self._evaluation_round = 0
        # Resolve tokenizer aliases exposed by the supported Work1 adapters.
        if hasattr(model, 'llama_tokenizer'):
            self.tokenizer = model.llama_tokenizer
        elif hasattr(model, 'tokenizer'):
            self.tokenizer = model.tokenizer
        elif hasattr(model, 'processor') and hasattr(model.processor, 'tokenizer'):
            self.tokenizer = model.processor.tokenizer
        else:
            raise AttributeError('Model does not expose a tokenizer')

        # Bigram LM 先验（预计算的 token 转移概率表）
        self._bigram_lp = None
        self._bigram_min_logprob = -20.0  # 未见 bigram 的默认低分
        if getattr(config, 'lm_prior_weight', 0.0) > 0:
            path = getattr(config, 'bigram_table_path', '')
            if path:
                self._load_bigram_table(path)

    # ================================================================
    #  核心工具方法
    # ================================================================

    def begin_evaluation_round(self):
        """Invalidate cached suffix losses for the next fixed-image search."""
        self._evaluation_round += 1
        self._suffix_loss_cache = {}

    @property
    def evaluation_round(self) -> int:
        """Identifier for losses evaluated on the current image and batch."""
        return self._evaluation_round

    def _build_text_prompts(self, adv_suffix_str: str, batch_queries: List[str],
                            text_prompt: str, batch_size: int) -> List[str]:
        """构建文本 prompt 列表"""
        separator = self.visible_suffix_separator + '<adv_split_start>'
        adv_split_end = '</adv_split_end>'
        return [
            text_prompt % (batch_queries[i] + separator + adv_suffix_str + adv_split_end)
            for i in range(batch_size)
        ]

    def evaluate_suffix(self, suffix_tokens: torch.Tensor, x_adv: torch.Tensor,
                        batch_queries: List[str], batch_targets: List[str],
                        text_prompt: str, batch_size: int,
                        prompt_wrapper_cls, attack_loss_fn) -> Tuple[float, float]:
        """
        评估一个完整后缀的 loss 和 reward（无梯度）。

        Args:
            suffix_tokens: 完整的对抗后缀 token 序列 (1D LongTensor)
            x_adv: 归一化后的对抗图像
            batch_queries: 查询文本列表
            batch_targets: 目标文本列表
            text_prompt: 带 %s 占位符的模板字符串
            batch_size: 批大小
            prompt_wrapper_cls: prompt_wrapper.Prompt 类
            attack_loss_fn: 兼容分支的 Attacker.attack_loss 方法。如果初始化
                searcher 时提供了 suffix_loss_fn，后缀评估优先使用后者。

        Returns:
            (loss_value, reward_value)
        """
        adv_suffix_str = self.tokenizer.decode(suffix_tokens, skip_special_tokens=True)
        text_prompts = self._build_text_prompts(adv_suffix_str, batch_queries, text_prompt, batch_size)

        # detach 图像输入，避免与外层计算图冲突
        x_adv_det = x_adv.detach()
        suffix_key = tuple(suffix_tokens.detach().cpu().tolist())

        if self._suffix_loss_cache is not None and suffix_key in self._suffix_loss_cache:
            loss = self._suffix_loss_cache[suffix_key]
            reward = self.config.loss_to_reward(loss)
            return loss, reward

        with torch.no_grad():
            if self.suffix_loss_fn is not None:
                loss = self.suffix_loss_fn(
                    x_adv_det,
                    batch_queries,
                    adv_suffix_str,
                    batch_targets,
                    text_prompt,
                ).item()
            else:
                prompt = prompt_wrapper_cls(
                    model=self.model,
                    text_prompts=text_prompts,
                    img_prompts=[[x_adv_det]],
                )
                prompt.img_embs = prompt.img_embs * batch_size
                prompt.update_cand_context_embs()
                loss = attack_loss_fn(prompt, batch_targets).item()

        if self._suffix_loss_cache is not None:
            self._suffix_loss_cache[suffix_key] = loss

        reward = self.config.loss_to_reward(loss)
        return loss, reward

    def evaluate_candidates_batch(self, candidates: List[SuffixCandidate],
                                   x_adv: torch.Tensor,
                                   batch_queries: List[str], batch_targets: List[str],
                                   text_prompt: str, batch_size: int,
                                   prompt_wrapper_cls, attack_loss_fn) -> List[Tuple[SuffixCandidate, float]]:
        """
        在当前批次上重新评估多个候选后缀。

        由于历史 loss 是在不同批次上计算的，此方法用于在当前批次上
        重新评估所有候选，返回按当前批次 loss 排序的结果。

        Args:
            candidates: SuffixCandidate 列表
            其余参数同 evaluate_suffix()

        Returns:
            results: (candidate, current_loss) 元组列表，按 loss 升序排列
        """
        results = []
        for cand in candidates:
            loss, reward = self.evaluate_suffix(
                cand.suffix_tokens, x_adv, batch_queries, batch_targets,
                text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn
            )
            cand.current_loss = loss
            cand.evaluation_round = self._evaluation_round
            if cand.node is not None:
                cand.node.update_value(
                    reward,
                    loss,
                    evaluation_round=self._evaluation_round,
                )
            results.append((cand, loss))

        # 按当前批次的 loss 排序
        results.sort(key=lambda x: x[1])
        return results

    def compute_position_gradient(self, suffix_tokens: torch.Tensor, x_adv: torch.Tensor,
                                   batch_queries: List[str], batch_targets: List[str],
                                   text_prompt: str, batch_size: int,
                                   prompt_wrapper_cls, attack_loss_fn) -> Tuple[torch.Tensor, float]:
        """
        计算当前后缀每个 token 位置的梯度。

        Args:
            suffix_tokens: 完整的对抗后缀 token 序列 (1D LongTensor)
            其余参数同 evaluate_suffix()

        Returns:
            text_grad: shape [suffix_len, vocab_size] 的梯度张量
            loss_val: 当前后缀的 loss 值
        """
        adv_suffix_str = self.tokenizer.decode(suffix_tokens, skip_special_tokens=True)
        text_prompts = self._build_text_prompts(adv_suffix_str, batch_queries, text_prompt, batch_size)

        # detach 并 clone 图像输入
        x_adv_det = x_adv.detach().clone().requires_grad_(True)

        prompt = prompt_wrapper_cls(
            model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv_det]]
        )
        prompt.img_embs = prompt.img_embs * batch_size
        prompt.update_context_embs()

        # 前向 + 反向
        self.model.zero_grad()
        loss = attack_loss_fn(prompt, batch_targets)
        loss.backward()

        # 聚合 batch 梯度
        text_grad = None
        for i in range(batch_size):
            new_grad = token_gradients(
                self.model, prompt.adv_suffix_tokens, prompt.adv_len,
                prompt.offset[i], prompt.context_embs[i].grad
            )
            if text_grad is None:
                text_grad = torch.zeros_like(new_grad)
            text_grad += new_grad

        self.model.zero_grad()
        loss_val = loss.item()

        del prompt, loss, x_adv_det
        gc.collect()

        return text_grad, loss_val

    def get_topk_tokens_for_position(self, text_grad: torch.Tensor, position: int,
                                      topk: int = None) -> torch.Tensor:
        """
        从梯度中提取指定位置的 Top-k 候选 token。
        梯度负方向最大的 token = 最能降低 loss 的 token。

        Args:
            text_grad: [suffix_len, vocab_size] 梯度张量
            position: 后缀中的位置索引 (0 ~ suffix_len-1)
            topk: 候选数量，默认使用 config.topk

        Returns:
            topk_tokens: 1D LongTensor, Top-k token ids
        """
        if topk is None:
            topk = self.config.topk
        
        # 防止索引越界
        safe_position = min(position, text_grad.shape[0] - 1)

        # text_grad 的负方向 = loss 下降最快的方向
        pos_grad = text_grad[safe_position]  # [vocab_size]
        # 取梯度最小（负方向最大）的 topk 个 token
        _, topk_indices = (-pos_grad).topk(topk)
        return topk_indices

    def _filter_semantic_candidates(self, token_ids: torch.Tensor, max_check: int = 256) -> torch.Tensor:
        """严格过滤：仅保留纯英文字符 token（可打印 ASCII + 至少含一个字母）。

        排除控制字符、非拉丁文字（中日韩/阿拉伯/西里尔等）、纯标点/符号。
        确保对抗后缀由可读的英文字符组成，避免乱码引起 VLM 警觉。

        关键修复：当 max_check 范围内英文 token 不足 min_keep 时，扩大搜索范围
        遍历全部 token_ids（topk=256），而非回退到未过滤的垃圾 token。
        """
        min_keep = 32

        def _is_english(tid):
            s = self.tokenizer.decode([tid])
            if not all(32 <= ord(c) <= 126 for c in s):
                return False
            if not any(c.isalpha() and c.isascii() for c in s):
                return False
            if not s.strip():
                return False
            # 第一个非空字符必须是英文字母（排除 $config, (ChatColor 等）
            stripped = s.lstrip()
            if not stripped or not (stripped[0].isalpha() and stripped[0].isascii()):
                return False
            return True

        # 第一轮：在 max_check 范围内过滤
        filtered = []
        for tid in token_ids[:max_check].tolist():
            if _is_english(tid):
                filtered.append(tid)

        # 若不足 min_keep，扩大搜索到全部候选（topk=256），而非回退未过滤
        if len(filtered) < min_keep:
            for tid in token_ids[max_check:].tolist():
                if _is_english(tid):
                    filtered.append(tid)
                    if len(filtered) >= min_keep:
                        break

        # 极端情况：全部 topk=256 中英文 token 也不足 min_keep
        # 保留已有的少量英文 token，宁缺毋滥
        if len(filtered) == 0:
            return token_ids[:1]
        return torch.tensor(filtered, device=token_ids.device)

    # ================================================================
    #  Embedding 空间采样（替代 GCG 梯度）
    # ================================================================

    def _get_embed_weights(self):
        """获取模型的 token embedding 权重矩阵 [vocab_size, hidden_dim]"""
        if hasattr(self.model, 'language_model'):
            return self.model.language_model.get_input_embeddings().weight
        elif hasattr(self.model, 'llama_model'):
            return self.model.llama_model.get_input_embeddings().weight
        elif hasattr(self.model, 'text_model'):
            return self.model.text_model.get_input_embeddings().weight
        else:
            raise AttributeError('Cannot find embedding weights in model')

    # ============================================================
    #  Bigram LM Prior
    # ============================================================

    def _load_bigram_table(self, path: str):
        """加载预计算的 bigram log-prob 表并构建反向索引。

        表格式: JSON dict, key="{tok_a}_{tok_b}", value=log_prob
        反向索引: dict, key=tok_a, value=[(tok_b, logprob), ...]
        """
        import json, os
        if not os.path.exists(path):
            print(f"[MCTS] WARNING: Bigram table not found at {path}, "
                  f"LM prior disabled")
            return
        try:
            with open(path, 'r') as f:
                self._bigram_lp = json.load(f)
            # 构建反向索引: prev_tok → [(next_tok, logprob), ...]
            self._bigram_index = {}
            for key, logprob in self._bigram_lp.items():
                parts = key.split('_', 1)
                if len(parts) != 2:
                    continue
                try:
                    prev = int(parts[0])
                    nxt = int(parts[1])
                except ValueError:
                    continue
                if prev not in self._bigram_index:
                    self._bigram_index[prev] = []
                self._bigram_index[prev].append((nxt, logprob))
            # 每个 prev 按 logprob 降序排列（一次性排序，O(1) 查表）
            for prev in self._bigram_index:
                self._bigram_index[prev].sort(key=lambda x: -x[1])
            if self.config.verbose:
                print(f"[MCTS] Loaded bigram LM prior: {len(self._bigram_lp)} entries, "
                      f"{len(self._bigram_index)} unique prev-tokens from {path}")
        except Exception as e:
            print(f"[MCTS] WARNING: Failed to load bigram table: {e}")

    def _score_by_bigram(self, candidate_ids: torch.Tensor,
                         prev_token_id: int) -> torch.Tensor:
        """查 bigram 表返回每个候选的 log-prob 分数。

        未见过的 bigram → 最低分 (self._bigram_min_logprob)

        Args:
            candidate_ids: 1D LongTensor of candidate token ids
            prev_token_id: 前一个位置的 token id

        Returns:
            scores: 1D FloatTensor, 每个候选的 bigram log-prob
        """
        if self._bigram_lp is None:
            return torch.zeros(len(candidate_ids), device=self.device)

        scores = torch.full(
            (len(candidate_ids),), self._bigram_min_logprob,
            device=self.device, dtype=torch.float
        )
        for i, tid in enumerate(candidate_ids.tolist()):
            key = f"{prev_token_id}_{tid}"
            scores[i] = self._bigram_lp.get(key, self._bigram_min_logprob)
        return scores

    def _get_top_bigram_continuations(self, prev_token_id: int,
                                        top_k: int = 128) -> List[int]:
        """从 bigram 表中找出最常跟随 prev_token_id 的 token 列表。

        使用加载时预计算的反向索引 → O(1) 查表。

        Args:
            prev_token_id: 前一个位置的 token id
            top_k: 返回的候选数上限

        Returns:
            continuation token id 列表（按频率降序排列）
        """
        if not hasattr(self, '_bigram_index') or self._bigram_index is None:
            return []
        matches = self._bigram_index.get(prev_token_id, [])
        if not matches:
            return []
        return [tok for tok, _ in matches[:top_k]]

    def _sample_candidates_by_embedding(
        self, current_token_id: int, num_candidates: int = 64,
        temperature: float = 1.0, uniform_ratio: float = 0.3,
        prev_token_id: int = None,
    ) -> torch.Tensor:
        """基于 embedding 空间采样候选 token（混合策略）。

        策略:
        - 若 current token 不含英文字母 → 100% 均匀采样（避免在 ! 等标点附近采样）
        - 若 current token 含英文字母 → embedding 相似度 + 均匀采样混合

        与 GCG 梯度的关键区别:
        - 不需要 backward pass → 省显存，省时间
        - 候选质量不依赖一阶近似的准确性
        - 均匀采样分量提供持续的探索能力

        Args:
            current_token_id: 当前位置的 token id
            num_candidates: 采样数量
            temperature: embedding 相似度 softmax 温度
            uniform_ratio: 均匀随机采样的比例

        Returns:
            sampled_ids: 1D LongTensor of candidate token ids
        """
        embed_weights = self._get_embed_weights()
        vocab_size = embed_weights.shape[0]

        # 检查 current token 是否是英文 token
        current_str = self.tokenizer.decode([current_token_id])
        current_is_english = (
            all(32 <= ord(c) <= 126 for c in current_str) and
            any(c.isalpha() and c.isascii() for c in current_str) and
            bool(current_str.strip())
        )

        if not current_is_english:
            # 当前 token 是 ! 等标点 — embedding 邻居也是标点，无意义
            # Bigram 引导采样: 若 bigram 表可用，从前一个 token 找最自然的接续
            if self._bigram_lp is not None and prev_token_id is not None:
                bg_tokens = self._get_top_bigram_continuations(
                    prev_token_id, top_k=num_candidates * 2
                )
                if len(bg_tokens) > 0:
                    result = torch.tensor(bg_tokens, device=self.device)
                    return self._filter_semantic_candidates(result, max_check=len(result))
            # Fallback: 均匀采样 + bigram 重排序
            uniform_ratio = 1.0

        num_embed_sample = max(1, int(num_candidates * (1 - uniform_ratio)))
        num_uniform_sample = num_candidates - num_embed_sample

        candidates = []
        seen_candidates = set()

        def add_candidate(tid: int) -> bool:
            if tid == current_token_id or tid in seen_candidates:
                return False
            seen_candidates.add(tid)
            candidates.append(tid)
            return True

        cos_sim = None
        if num_embed_sample > 0:
            # ---- 分量 1: embedding 余弦相似度采样（局部搜索） ----
            current_emb = embed_weights[current_token_id]  # [D]
            cos_sim = F.cosine_similarity(
                current_emb.unsqueeze(0), embed_weights, dim=1
            )  # [vocab_size]
            probs = torch.softmax(cos_sim / max(temperature, 0.01), dim=0)
            probs[current_token_id] = 0.0
            probs = probs / probs.sum()

            embed_samples = torch.multinomial(
                probs, num_samples=min(num_embed_sample * 3, vocab_size - 1),
                replacement=False
            )
            for tid in embed_samples.tolist():
                if len(candidates) >= num_embed_sample:
                    break
                add_candidate(tid)

        # ---- 分量 2: 均匀随机采样（全局探索） ----
        # 直接随机采样 + 碰撞跳过，避免创建 152k 的 list；保留追加顺序。
        uniform_added = 0
        max_attempts = num_uniform_sample * 10  # 防止极端情况死循环
        attempts = 0
        while uniform_added < num_uniform_sample and attempts < max_attempts:
            tid = random.randint(0, vocab_size - 1)
            if add_candidate(tid):
                uniform_added += 1
            attempts += 1

        # ---- 过滤 + Bigram LM 先验重排序 ----
        result = torch.tensor(candidates, device=self.device)
        result = self._filter_semantic_candidates(result, max_check=len(result))

        # 使用真实 embedding 余弦相似度与 Bigram LM 先验重排序。
        lm_w = getattr(self.config, 'lm_prior_weight', 0.0)
        if len(result) > 1:
            if cos_sim is not None:
                embed_scores = cos_sim[result].float()
                emin, emax = embed_scores.min(), embed_scores.max()
                embed_norm = (embed_scores - emin) / (emax - emin + 1e-8)
            else:
                embed_norm = torch.zeros(len(result), device=self.device)

            has_bigram = lm_w > 0 and prev_token_id is not None
            if has_bigram:
                bigram_scores = self._score_by_bigram(result, prev_token_id).float()
                bmin, bmax = bigram_scores.min(), bigram_scores.max()
                bigram_norm = (bigram_scores - bmin) / (bmax - bmin + 1e-8)
                combined = (1.0 - lm_w) * embed_norm + lm_w * bigram_norm
            else:
                combined = embed_norm

            order = combined.argsort(descending=True)
            result = result[order]

        return result

    def _greedy_evaluate_and_select(
        self, candidate_tokens: torch.Tensor, current_node: AdvTreeNode,
        x_adv: torch.Tensor, batch_queries: List[str], batch_targets: List[str],
        text_prompt: str, batch_size: int, prompt_wrapper_cls, attack_loss_fn,
        initial_eval: int = 3, max_eval: int = 8,
        expand_threshold: float = 0.01,
    ) -> Tuple[int, float, float]:
        """评估多个候选 token，返回实际 loss 最低的那个。

        这是"evaluate-then-select"策略的核心：
        - 不信任任何梯度/采样信号，直接评估实际 loss
        - 消除了 GCG 梯度的"假更新"问题
        - 不创建树节点，直接用 build 临时后缀评估，避免污染树结构

        Args:
            candidate_tokens: 候选 token id 列表
            current_node: 当前树节点
            initial_eval: 首轮评估的候选数
            max_eval: 首轮没有明显改善时最多评估的候选数
            expand_threshold: 相对当前 token 的最小 loss 改善
            其余参数同 evaluate_suffix()

        Returns:
            (best_token_id, best_loss, best_reward)
        """
        # 构建基础后缀（candidate 之前的部分由路径决定）
        base_suffix = current_node.build_full_suffix()
        eval_pos = current_node.next_position

        baseline_loss, baseline_reward = self.evaluate_suffix(
            base_suffix, x_adv, batch_queries, batch_targets,
            text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn
        )
        best_tok = base_suffix[eval_pos].item()
        best_loss = baseline_loss
        best_reward = baseline_reward

        candidate_count = len(candidate_tokens)
        initial_end = min(initial_eval, max_eval, candidate_count)

        def evaluate_range(start: int, end: int):
            nonlocal best_tok, best_loss, best_reward
            for tok_id in candidate_tokens[start:end].tolist():
                test_suffix = base_suffix.clone()
                test_suffix[eval_pos] = tok_id
                test_loss, test_reward = self.evaluate_suffix(
                    test_suffix, x_adv, batch_queries, batch_targets,
                    text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn
                )
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_reward = test_reward
                    best_tok = tok_id

        evaluate_range(0, initial_end)
        evaluated = initial_end
        if (baseline_loss - best_loss < expand_threshold and
                initial_end < min(max_eval, candidate_count)):
            evaluated = min(max_eval, candidate_count)
            evaluate_range(initial_end, evaluated)

        if self.config.verbose:
            print(f"[MCTS] Greedy eval: candidates={evaluated}, "
                  f"baseline={baseline_loss:.4f}, best={best_loss:.4f}")

        return best_tok, best_loss, best_reward

    # ================================================================
    #  MCTS 四阶段
    # ================================================================

    def select_node(self, root: AdvTreeNode) -> AdvTreeNode:
        """
        Phase 1: Selection
        沿 UCB 最大的子节点向下遍历，直到：
        - 找到未完全扩展的节点，或
        - 到达最大深度（叶节点）
        """
        node = root
        while node.children and not node.is_leaf_depth:
            best_child = self._get_best_child_ucb(node)
            if best_child is None:
                # 虚拟动作胜出 → 在此节点扩展新分支
                break
            node = best_child
        return node

    def _get_best_child_ucb(self, node: AdvTreeNode) -> AdvTreeNode:
        cfg = self.config
        best_value = float('-inf')
        best_nodes = []
        
        for child in node.children.values():
            # 微小随机抖动 (1e-6 量级) 防止相同 V 值下总是选择同一子节点
            jitter = random.uniform(0, 1e-6)
            ucb = child.V + cfg.exploration_constant * math.sqrt(
                    2 * math.log(node.numVisits + 1) / (child.numVisits + 1)
            ) + jitter
            if ucb > best_value:
                best_value = ucb
                best_nodes = [child]
            elif ucb == best_value:
                best_nodes.append(child)
        
        # 渐进加宽 (Progressive Widening):
        # 允许的子节点数 = max(1, ceil(numVisits ^ pw_alpha))
        # 只有当子节点数 < pw_limit 时，虚拟动作才可用
        pw_alpha = getattr(cfg, 'pw_alpha', 0.5)
        pw_limit = max(1, int(math.ceil(node.numVisits ** pw_alpha)))
        
        has_unexpanded = False
        if not node.is_leaf_depth and len(node.children) < pw_limit:
            if node.all_candidates_pool is None:
                has_unexpanded = True
            elif node.unexpanded_candidates or node.pool_index < len(node.all_candidates_pool):
                has_unexpanded = True
        
        if has_unexpanded:
            # 虚拟动作 exploitation: 用子节点实际 V 的均值
            if node.children:
                child_values = [c.V for c in node.children.values()]
                virtual_exploit = sum(child_values) / len(child_values)
            else:
                virtual_exploit = node.V  # 首次扩展，退回父节点 V
            
            # 探索项：以子节点数为有效访问次数
            n_virtual = len(node.children)
            virtual_explore = cfg.exploration_constant * math.sqrt(
                2 * math.log(node.numVisits + 1) / (n_virtual + 1)
            )
            
            virtual_ucb = virtual_exploit + virtual_explore
            
            if virtual_ucb > best_value:
                return None
            elif virtual_ucb == best_value:
                best_nodes.append(None)

        result = random.choice(best_nodes)
        return result

    def expand(self, node: AdvTreeNode, x_adv: torch.Tensor,
           batch_queries: List[str], batch_targets: List[str],
           text_prompt: str, batch_size: int,
           prompt_wrapper_cls, attack_loss_fn,
           current_iteration: int = -1) -> AdvTreeNode:
        cfg = self.config

        if node.is_leaf_depth:
            return node

        use_embed_sampling = getattr(cfg, 'use_embedding_sampling', False)

        # 首次扩展：初始化候选队列
        if node.all_candidates_pool is None:
            if use_embed_sampling:
                # ---- Embedding 空间采样（无需 backward pass） ----
                current_suffix = node.build_full_suffix()
                expand_position = node.next_position
                current_tok = current_suffix[expand_position].item()

                embed_temp = getattr(cfg, 'embedding_sample_temperature', 1.0)
                uniform_ratio = getattr(cfg, 'embedding_uniform_ratio', 0.3)
                num_samples = getattr(cfg, 'embedding_num_samples', 64)

                # 前一个位置的 token，用于 bigram LM 先验
                prev_tok = current_suffix[expand_position - 1].item() \
                    if expand_position > 0 else None

                topk_tokens = self._sample_candidates_by_embedding(
                    current_tok, num_candidates=num_samples,
                    temperature=embed_temp, uniform_ratio=uniform_ratio,
                    prev_token_id=prev_tok,
                )
                if cfg.verbose:
                    print(f"[MCTS] Embed sampling at pos={expand_position}, "
                          f"current_tok={current_tok}('{self.tokenizer.decode([current_tok])}'), "
                          f"got {len(topk_tokens)} candidates")
            else:
                # ---- 原始 GCG 梯度路径（保留兼容） ----
                current_suffix = node.build_full_suffix()
                expand_position = node.next_position

                text_grad, _ = self.compute_position_gradient(
                    current_suffix, x_adv, batch_queries, batch_targets,
                    text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn
                )
                topk_tokens = self.get_topk_tokens_for_position(text_grad, expand_position)
                del text_grad
                gc.collect()

                # 语义过滤
                raw_count = len(topk_tokens)
                topk_tokens = self._filter_semantic_candidates(topk_tokens, cfg.topk)
                if cfg.verbose:
                    print(f"[MCTS] Token filter: {raw_count} → {len(topk_tokens)} candidates "
                          f"(removed {raw_count - len(topk_tokens)} low-quality)")

            node.all_candidates_pool = topk_tokens[:cfg.max_branch].tolist()
            initial_count = min(cfg.branch, len(node.all_candidates_pool))

            node.unexpanded_candidates = list(node.all_candidates_pool[:initial_count])
            node.pool_index = initial_count

        # 优先从活跃队列弹出
        if not node.unexpanded_candidates:
            # 活跃队列空了，从后备池补 1 个（progressive widening）
            if node.pool_index < len(node.all_candidates_pool):
                node.unexpanded_candidates = [node.all_candidates_pool[node.pool_index]]
                node.pool_index += 1
                if cfg.verbose:
                    print(f"[MCTS] Widening at depth {node.depth}: unlocked rank {node.pool_index}")
            else:
                return node  # 所有候选已用完

        # 弹出 1 个候选，跳过已存在的
        tok_id = None
        while node.unexpanded_candidates:
            candidate = node.unexpanded_candidates.pop(0)
            if candidate not in node.children:
                tok_id = candidate
                break

        if tok_id is None:
            return node

        child = node.append_child(tok_id)
        child_suffix = child.build_full_suffix()
        child_loss, child_reward = self.evaluate_suffix(
            child_suffix, x_adv, batch_queries, batch_targets,
            text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn
        )
        child.update_value(
            child_reward,
            child_loss,
            iteration=current_iteration,
            evaluation_round=self._evaluation_round,
        )

        torch.cuda.empty_cache()
        return child
    
    def greedy_rollout(self, node: AdvTreeNode, x_adv: torch.Tensor,
                       batch_queries: List[str], batch_targets: List[str],
                       text_prompt: str, batch_size: int,
                       prompt_wrapper_cls, attack_loss_fn,
                       current_iteration: int = -1) -> Tuple[float, Optional[AdvTreeNode]]:
        """
        Phase 3: Greedy MC Rollout（evaluate-then-select 版本）

        从当前节点向下前进 roll_forward_steps 步。
        每步：embedding 空间采样候选 → 逐 token 评估实际 loss → 选最优 token → 创建子节点。

        与 GCG 梯度版本的关键区别:
        - 不计算梯度（无 backward pass），而是直接评估候选的 loss
        - 候选选择基于实际 loss 而非一阶近似 → 消除"假更新"
        - 评估多个候选 (greedy_eval_candidates) 而非只取梯度 top-1

        Returns:
            max_reward: rollout 过程中探测到的最高奖励值
            deepest_node: rollout 到达的最深节点
        """
        cfg = self.config
        use_embed_sampling = getattr(cfg, 'use_embedding_sampling', False)
        num_eval = getattr(cfg, 'greedy_eval_candidates', 3)
        max_eval = getattr(cfg, 'greedy_eval_candidates_max', 8)
        expand_threshold = getattr(cfg, 'greedy_expand_threshold', 0.01)
        embed_temp = getattr(cfg, 'embedding_sample_temperature', 1.0)
        uniform_ratio = getattr(cfg, 'embedding_uniform_ratio', 0.3)
        num_samples = getattr(cfg, 'embedding_num_samples', 64)

        current_node = node
        max_reward = node.V
        suffix_len = node.suffix_len

        for step in range(cfg.roll_forward_steps):
            if current_node.depth >= suffix_len:
                break
            rollout_pos = current_node.next_position

            current_suffix = current_node.build_full_suffix()

            if use_embed_sampling:
                # ---- Embedding 采样 + evaluate-then-select ----
                current_tok = current_suffix[rollout_pos].item()
                prev_tok = current_suffix[rollout_pos - 1].item() \
                    if rollout_pos > 0 else None
                candidate_pool = self._sample_candidates_by_embedding(
                    current_tok, num_candidates=num_samples,
                    temperature=embed_temp, uniform_ratio=uniform_ratio,
                    prev_token_id=prev_tok,
                )
                selected_token_id, selected_loss, selected_reward = self._greedy_evaluate_and_select(
                    candidate_pool, current_node, x_adv,
                    batch_queries, batch_targets, text_prompt,
                    batch_size, prompt_wrapper_cls, attack_loss_fn,
                    initial_eval=num_eval, max_eval=max_eval,
                    expand_threshold=expand_threshold,
                )
                # 缓存采样结果到候选池（供后续 expand 复用）
                if current_node.all_candidates_pool is None:
                    current_node.all_candidates_pool = candidate_pool[:cfg.max_branch].tolist()
                    current_node.pool_index = 0
            else:
                # ---- 原始 GCG 梯度路径 ----
                text_grad, current_loss = self.compute_position_gradient(
                    current_suffix, x_adv, batch_queries, batch_targets,
                    text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn
                )
                topk_candidates = self.get_topk_tokens_for_position(text_grad, rollout_pos, topk=cfg.topk)
                topk_filtered = self._filter_semantic_candidates(topk_candidates, cfg.topk)
                selected_token_id = topk_filtered[0].item()

                if current_node.all_candidates_pool is None:
                    current_node.all_candidates_pool = topk_filtered.tolist()
                    current_node.pool_index = 0

                del text_grad
                gc.collect()

            # 将贪婪选择的 token 作为子节点加入树（如果不存在）
            if selected_token_id not in current_node.children:
                child = current_node.append_child(selected_token_id)
            else:
                child = current_node.children[selected_token_id]

            # Embedding 路径已在候选选择时评估；梯度路径仍需在此评估。
            if use_embed_sampling:
                child_loss, child_reward = selected_loss, selected_reward
            else:
                child_suffix = child.build_full_suffix()
                child_loss, child_reward = self.evaluate_suffix(
                    child_suffix, x_adv, batch_queries, batch_targets,
                    text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn
                )
            child.update_value(
                child_reward,
                child_loss,
                iteration=current_iteration,
                evaluation_round=self._evaluation_round,
            )

            current_node = child
            if current_node.V > max_reward:
                max_reward = current_node.V

        torch.cuda.empty_cache()
        return max_reward, current_node

    def backpropagate(self, node: AdvTreeNode, rollout_value: float):
        """
        Phase 4: Backpropagation
        将 rollout 得到的最高奖励沿路径反向传播至根节点。

        支持两种策略：
        - 'max': Max-Backup，保留路径上的最高奖励
        - 'weighted_avg': 加权平均平滑更新
        """
        cfg = self.config
        current = node

        while current is not None:
            current.numVisits += 1

            if cfg.backup_strategy == 'mean':
                # 子节点访问加权平均
                if current.children:
                    child_Vs = [c.V * c.numVisits for c in current.children.values() if c.numVisits > 0]
                    total_visits = sum(c.numVisits for c in current.children.values() if c.numVisits > 0)
                    if total_visits > 0:
                        current.V = sum(child_Vs) / total_visits
                else:
                    # 叶节点：用 rollout_value 更新
                    current.V = (current.V * (current.numVisits - 1) + rollout_value) / current.numVisits

            elif cfg.backup_strategy == 'max':
                current.V = max(current.V, rollout_value)

            else:  # weighted_avg
                current.V = cfg.alpha * current.V + (1 - cfg.alpha) * rollout_value

            # best_loss 仍用 min 追踪（这是全局信息，不影响 UCB 选择）
            if rollout_value > 1e-10:
                implied_loss = -math.log(rollout_value) / cfg.reward_beta
                current.best_loss = min(current.best_loss, implied_loss)

            current = current.parent

    # ================================================================
    #  完整搜索入口
    # ================================================================

    def search(self, adv_suffix_tokens_init: torch.Tensor, x_adv: torch.Tensor,
               batch_queries: List[str], batch_targets: List[str],
               text_prompt: str, batch_size: int,
               prompt_wrapper_cls, attack_loss_fn) -> Tuple[List[SuffixCandidate], AdvTreeNode]:
        """
        完整 MCTS 逐位置搜索。

        搜索树结构:
            Root (depth=0, 初始后缀)
            ├── depth=1: position 0 的候选 tokens
            │   ├── depth=2: position 1 的候选 tokens
            │   │   └── ...
            │   └── ...
            └── ...

        每次迭代选择一条路径，对该路径上第一个未扩展的位置进行扩展，
        然后从该位置向后贪婪 rollout 评估攻击潜力。

        重要改进：
        - 不使用早停策略，运行完所有迭代
        - 返回多个候选后缀而非单一"最优"
        - 由调用方在当前批次上重新评估候选

        Args:
            adv_suffix_tokens_init: 初始对抗后缀 (1D LongTensor, 长度=suffix_len)
            x_adv: 归一化后的对抗图像
            batch_queries: 查询文本列表
            batch_targets: 目标文本列表
            text_prompt: 模板字符串
            batch_size: 批大小
            prompt_wrapper_cls: prompt_wrapper.Prompt 类
            attack_loss_fn: Attacker.attack_loss 方法

        Returns:
            candidates: SuffixCandidate 列表（按历史 loss 排序，仅供参考）
            root: 搜索树根节点（可用于可视化或进一步分析）
        """
        cfg = self.config

        # 创建根节点：存储初始后缀
        root = AdvTreeNode(
            token_id=-1,
            parent=None,
            depth=0,
            base_suffix_tokens=adv_suffix_tokens_init
        )

        # 评估根节点（初始后缀）
        root_loss, root_reward = self.evaluate_suffix(
            adv_suffix_tokens_init, x_adv, batch_queries, batch_targets,
            text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn
        )
        root.update_value(
            root_reward,
            root_loss,
            iteration=0,
            evaluation_round=(self._evaluation_round or None),
        )
        root.numVisits = 1

        if cfg.verbose:
            print(f"[MCTS] Root: loss={root_loss:.4f}, reward={root_reward:.4f}, "
                  f"suffix_len={adv_suffix_tokens_init.shape[0]}")

        for iteration in range(cfg.iteration_limit):
            if cfg.verbose:
                print(f"\n[MCTS] === Iteration {iteration + 1}/{cfg.iteration_limit} ===")

            # Phase 1: Selection
            leaf = self.select_node(root)
            if cfg.verbose:
                print(f"[MCTS] Selected: depth={leaf.depth}, pos={leaf.position}, "
                      f"V={leaf.V:.4f}, loss={leaf.best_loss:.4f}")

            # 已达最大深度（所有位置都已确定）
            if leaf.is_leaf_depth:
                if cfg.verbose:
                    print(f"[MCTS] Reached max depth ({leaf.depth}), backpropagating.")
                self.backpropagate(leaf, leaf.V)
                continue

            # Phase 2: Expansion
            expanded_node = self.expand(
                leaf, x_adv, batch_queries, batch_targets,
                text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn,
                current_iteration=iteration
            )

            if not leaf.children:
                if cfg.verbose:
                    print("[MCTS] No candidates generated, skipping.")
                continue

            # 优先从刚扩展出的真实 child 做 rollout；若没有新 child，则回退到最优子节点。
            best_child = expanded_node if expanded_node is not leaf else leaf.get_best_child_by_value()
            if best_child is None:
                continue

            if cfg.verbose:
                print(f"[MCTS] Best child: pos={best_child.position}, "
                      f"tok_id={best_child.token_id}, V={best_child.V:.4f}, "
                      f"loss={best_child.best_loss:.4f}")

            # Phase 3: Greedy Rollout
            rollout_max_V, deepest_node = self.greedy_rollout(
                best_child, x_adv, batch_queries, batch_targets,
                text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn,
                current_iteration=iteration
            )
            best_child.rollout_max_V = rollout_max_V

            if cfg.verbose:
                print(f"[MCTS] Rollout: max_reward={rollout_max_V:.4f}, "
                      f"reached_depth={deepest_node.depth if deepest_node else 'N/A'}")

            # Phase 4: Backpropagation
            # 从 rollout 最深节点开始回传
            backprop_node = deepest_node if deepest_node else best_child
            self.backpropagate(backprop_node, rollout_max_V)

            # 打印搜索树统计
            if cfg.verbose:
                stats = root.get_tree_stats()
                print(f"[MCTS] Tree: nodes={stats['total_nodes']}, "
                      f"max_depth={stats['max_depth']}, "
                      f"complete_suffixes={stats['complete_suffixes']}, "
                      f"min_loss={stats['min_loss']:.4f}")

        # 搜索结束，收集候选后缀
        # 优先收集完整后缀（深度 = suffix_len）
        complete_candidates = root.collect_complete_suffixes(
            self.tokenizer, top_k=cfg.num_candidates_return
        )

        # 如果完整后缀不够，补充有潜力的部分后缀
        if len(complete_candidates) < cfg.num_candidates_return:
            # 动态调整 min_depth：取当前树最大深度的一半，至少为 1
            stats = root.get_tree_stats()
            adaptive_min_depth = max(1, stats['max_depth'] // 2)
            
            promising_candidates = root.collect_promising_nodes(
                self.tokenizer,
                top_k=cfg.num_candidates_return - len(complete_candidates),
                min_depth=adaptive_min_depth  # 使用自适应的最小深度
            )
            complete_candidates.extend(promising_candidates)

        if cfg.verbose:
            print(f"\n[MCTS] Search complete.")
            print(f"[MCTS] Collected {len(complete_candidates)} candidate suffixes.")
            for i, cand in enumerate(complete_candidates[:3]):
                print(f"[MCTS]   #{i+1}: depth={cand.depth}, "
                      f"hist_loss={cand.historical_loss:.4f}, "
                      f"suffix='{cand.suffix_str[:50]}...'")

        return complete_candidates, root

    def search_continue(self, root: AdvTreeNode, x_adv: torch.Tensor,
                        batch_queries: List[str], batch_targets: List[str],
                        text_prompt: str, batch_size: int,
                        prompt_wrapper_cls, attack_loss_fn) -> List[SuffixCandidate]:
        """
        在现有树上继续搜索（不创建新树）。
        """
        cfg = self.config

        if cfg.verbose:
            stats = root.get_tree_stats()
            print(f"[MCTS] Continuing search: {stats['total_nodes']} nodes, "
                  f"max_depth={stats['max_depth']}")

        for iteration in range(cfg.iteration_limit):
            if cfg.verbose:
                print(f"\n[MCTS] === Iteration {iteration + 1}/{cfg.iteration_limit} ===")

            # Phase 1: Selection
            leaf = self.select_node(root)
            if cfg.verbose:
                print(f"[MCTS] Selected: depth={leaf.depth}, V={leaf.V:.4f}")

            if leaf.is_leaf_depth:
                leaf_suffix = leaf.build_full_suffix()
                fresh_loss, fresh_reward = self.evaluate_suffix(
                    leaf_suffix, x_adv, batch_queries, batch_targets,
                    text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn
                )
                leaf.update_value(
                    fresh_reward,
                    fresh_loss,
                    iteration=iteration,
                    evaluation_round=self._evaluation_round,
                )
                self.backpropagate(leaf, fresh_reward)
                continue

            # Phase 2: Expansion
            new_or_leaf_node = self.expand(
                leaf, x_adv, batch_queries, batch_targets,
                text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn,
                current_iteration=iteration
            )

            if not leaf.children:
                continue
            
            if new_or_leaf_node != leaf:
                best_child = new_or_leaf_node
            else:
                best_child = leaf.get_best_child_by_value()
                if getattr(best_child, 'has_been_rolled_out', False):
                    self.backpropagate(best_child, best_child.V)
                    continue

            if cfg.verbose:
                print(f"[MCTS] Best child: pos={best_child.position}, "
                      f"tok_id={best_child.token_id}, V={best_child.V:.4f}")

            # Phase 3: Rollout
            rollout_max_V, deepest_node = self.greedy_rollout(
                best_child, x_adv, batch_queries, batch_targets,
                text_prompt, batch_size, prompt_wrapper_cls, attack_loss_fn,
                current_iteration=iteration
            )
            best_child.has_been_rolled_out = True
            # Phase 4: Backpropagation
            backprop_node = deepest_node if deepest_node else best_child
            self.backpropagate(backprop_node, rollout_max_V)

            if cfg.verbose:
                stats = root.get_tree_stats()
                print(f"[MCTS] Tree: nodes={stats['total_nodes']}, "
                      f"max_depth={stats['max_depth']}")

        # 本轮节点是主候选，历史节点只负责召回。调用方会在
        # 当前图像和 batch 上统一重评，历史 loss 不参与最终排序。
        candidate_limit = max(0, cfg.num_candidates_return)
        current_round = self._evaluation_round or None
        fresh_all = root.collect_best_candidates(
            self.tokenizer,
            top_k=candidate_limit,
            evaluation_round=current_round,
        )
        min_depth = cfg.normalized_min_accept_depth(root.suffix_len)
        deep_quota = candidate_limit // 2 if min_depth > 0 else 0
        fresh_deep = root.collect_deep_candidates(
            self.tokenizer,
            top_k=deep_quota,
            min_depth=min_depth,
            evaluation_round=current_round,
        )

        fresh_candidates = []
        seen_suffixes = set()
        overall_quota = candidate_limit - deep_quota
        fresh_ordered = (
            fresh_all[:overall_quota]
            + fresh_deep
            + fresh_all[overall_quota:]
        )
        for candidate in fresh_ordered:
            suffix_key = tuple(candidate.suffix_tokens.tolist())
            if suffix_key in seen_suffixes:
                continue
            fresh_candidates.append(candidate)
            seen_suffixes.add(suffix_key)
            if len(fresh_candidates) >= candidate_limit:
                break

        historical_quota = max(0, cfg.historical_candidate_quota)
        historical_candidates = []
        if historical_quota > 0:
            # 多取一些再排除本轮节点和重复后缀，避免历史配额被重叠项吃掉。
            history_scan_limit = candidate_limit + historical_quota * 3
            historical_all = root.collect_best_candidates(
                self.tokenizer,
                top_k=history_scan_limit,
                evaluation_round=None,
                exclude_evaluation_round=current_round,
            )
            historical_deep = root.collect_deep_candidates(
                self.tokenizer,
                top_k=history_scan_limit,
                min_depth=min_depth,
                evaluation_round=None,
                exclude_evaluation_round=current_round,
            )
            history_deep_quota = historical_quota // 2
            history_ordered = (
                historical_all[:historical_quota - history_deep_quota]
                + historical_deep[:history_deep_quota]
                + historical_all[historical_quota - history_deep_quota:]
                + historical_deep[history_deep_quota:]
            )
            for candidate in history_ordered:
                suffix_key = tuple(candidate.suffix_tokens.tolist())
                if suffix_key in seen_suffixes:
                    continue
                historical_candidates.append(candidate)
                seen_suffixes.add(suffix_key)
                if len(historical_candidates) >= historical_quota:
                    break

        candidates = fresh_candidates + historical_candidates

        if cfg.verbose:
            print(f"[MCTS] Candidate recall: fresh={len(fresh_candidates)}, "
                  f"historical={len(historical_candidates)}, "
                  f"returned={len(candidates)}, "
                  f"round={self._evaluation_round}")
            for i, candidate in enumerate(candidates[:3]):
                print(f"[MCTS]   recall #{i + 1}: source={candidate.recall_source}, "
                      f"loss={candidate.current_loss:.4f}, depth={candidate.depth}")

        return candidates
