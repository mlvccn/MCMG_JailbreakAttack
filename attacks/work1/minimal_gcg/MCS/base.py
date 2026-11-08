import torch
import math
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SuffixCandidate:
    """对抗后缀候选结果"""
    suffix_tokens: torch.Tensor  # 完整后缀 token 序列
    suffix_str: str              # 解码后的字符串
    depth: int                   # 来自树的深度
    historical_loss: float       # 历史批次上的 loss（仅参考）
    historical_V: float          # 历史批次上的 V（仅参考）
    path_tokens: List[int]       # 从根到该节点的路径
    node: object = None          # 对应的 AdvTreeNode（用于修剪）
    current_loss: float = float('inf')  # 当前评估轮次的真实 loss
    evaluation_round: int = -1          # current_loss 所属的评估轮次
    recall_source: str = 'unknown'      # 候选召回来源，不参与最终排序


class AdvTreeNode(object):
    """
    MCTS 树节点 —— 逐位置对抗后缀搜索。

    树结构定义：
    - 根节点 (depth=0)：存储初始对抗后缀的完整 token 序列作为基准
    - 根节点通过 start_offset 指定本轮首先优化的位置
    - 子节点 (depth=d)：对应位置 (start_offset + d - 1) % suffix_len
    - 从根到深度为 d 的节点路径，表示循环位置顺序中的前 d 个选择
    - 最大深度 = suffix_len（通常为 20），总深度含根节点 = suffix_len + 1

    重要说明：
    - MCTS 使用固定评估 batch，但图像会在搜索轮次之间变化，因此历史
      V/best_loss 只用于指导搜索，不能用于当前轮候选排序
    - current_loss 仅在对应 evaluation_round 内可比

    示例（suffix_len=4）：
        Root(depth=0): [tok_0, tok_1, tok_2, tok_3]  （初始后缀）
         ├── Child_A(depth=1): pos=0 替换为 tok_A  → [tok_A, tok_1, tok_2, tok_3]
         │    ├── Child_X(depth=2): pos=1 替换为 tok_X → [tok_A, tok_X, tok_2, tok_3]
         │    └── Child_Y(depth=2): pos=1 替换为 tok_Y → [tok_A, tok_Y, tok_2, tok_3]
         └── Child_B(depth=1): pos=0 替换为 tok_B  → [tok_B, tok_1, tok_2, tok_3]
    """

    def __init__(self, token_id: int = -1, parent: Optional['AdvTreeNode'] = None,
                 depth: int = 0, base_suffix_tokens: torch.Tensor = None,
                 start_offset: int = 0):
        """
        Args:
            token_id: 当前节点选择的 token id。
                      根节点为 -1（不代表任何替换）。
                      非根节点的实际位置由 start_offset 和 depth 共同决定。
            parent: 父节点
            depth: 节点深度。0 = 根节点, d = 已确定了前 d 个位置的替换
            base_suffix_tokens: 仅根节点需要设置，存储初始的完整后缀 tokens (1D LongTensor)
            start_offset: 仅根节点使用，本轮循环位置顺序的起点
        """
        self.token_id = token_id
        self.parent = parent
        self.depth = depth

        # 仅根节点持有，子节点通过 _get_root() 获取
        self._base_suffix_tokens = base_suffix_tokens.clone() if base_suffix_tokens is not None else None
        self._start_offset = int(start_offset) if parent is None else None

        # ---------- MCTS 统计量 ----------
        self.numVisits = 0
        self.V = 0.0               # 节点价值（奖励，越高越好）- 注意：跨批次不可比
        self.best_loss = float('inf')  # 该路径下探测到的最低 loss - 注意：跨批次不可比
        self.current_loss = float('inf')  # 当前图像和固定 batch 上的最新真实 loss
        self.evaluation_round = -1       # current_loss 所属的 MCTS 评估轮次

        # ---------- 子节点管理 ----------
        # key: token_id (int), value: AdvTreeNode
        self.children = {}
        # self.isFullyExpanded = False

        # rollout 过程中探测到的最大奖励
        self.rollout_max_V = 0.0
        self.has_been_rolled_out = False
        # 记录该节点被评估的迭代轮次（用于判断新鲜度）
        self.last_eval_iteration = -1

        # 候选队列
        self.unexpanded_candidates = None
        self.all_candidates_pool = None
        self.candidate_index = 0
        self.pool_index = 0

    # ================================================================
    #  属性与工具方法
    # ================================================================

    @property
    def suffix_len(self) -> int:
        """后缀总长度"""
        root = self._get_root()
        return root._base_suffix_tokens.shape[0]

    @property
    def position(self) -> int:
        """
        当前节点对应的后缀位置索引。
        根节点返回 -1；其他节点按根节点的循环起点映射。
        """
        if self.is_root:
            return -1
        return self.position_for_depth(self.depth)

    @property
    def start_offset(self) -> int:
        """返回当前树规范化后的循环起点。"""
        root = self._get_root()
        return root._start_offset % self.suffix_len

    @property
    def next_position(self) -> int:
        """返回从当前节点继续扩展时对应的真实 token 位置。"""
        if self.is_leaf_depth:
            return -1
        return self.position_for_depth(self.depth + 1)

    def position_for_depth(self, depth: int) -> int:
        """将非根节点深度映射到后缀中的真实位置。"""
        if depth <= 0:
            return -1
        return (self.start_offset + depth - 1) % self.suffix_len

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf_depth(self) -> bool:
        """是否已达最大深度（所有位置都已确定）"""
        return self.depth >= self.suffix_len

    @property
    def is_complete_suffix(self) -> bool:
        """是否是完整的对抗后缀（深度 = suffix_len）"""
        return self.depth == self.suffix_len

    def _get_root(self) -> 'AdvTreeNode':
        """向上回溯获取根节点"""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def get_path_tokens(self) -> List[int]:
        """
        获取从根到当前节点路径上所有非根节点选择的 token id 列表。
        返回列表长度 = self.depth，其位置由 get_path_positions() 给出。
        """
        path = []
        node = self
        while node.parent is not None:
            path.append(node.token_id)
            node = node.parent
        path.reverse()
        return path

    def get_path_positions(self) -> List[int]:
        """返回路径上每个 token 决策对应的真实后缀位置。"""
        return [self.position_for_depth(depth) for depth in range(1, self.depth + 1)]

    def build_full_suffix(self) -> torch.Tensor:
        """
        根据从根到当前节点的路径，构建当前的完整对抗后缀 token 序列。

        对于深度 d 的节点，循环顺序中的前 d 个位置由路径决定，
        其他位置保持根节点 baseline。

        Returns:
            suffix_tokens: 1D LongTensor, 完整的对抗后缀 tokens
        """
        root = self._get_root()
        suffix = root._base_suffix_tokens.clone()

        # 获取路径上的 token 替换决策
        path_tokens = self.get_path_tokens()

        # 将路径上的决策应用到对应位置
        for pos, tok_id in zip(self.get_path_positions(), path_tokens):
            suffix[pos] = tok_id

        return suffix

    # ================================================================
    #  子节点管理
    # ================================================================

    def append_child(self, token_id: int) -> 'AdvTreeNode':
        """
        为当前节点添加子节点，表示在 next_position 处选择 token_id。

        Args:
            token_id: 下一个位置选择的 token id
        Returns:
            新创建的子节点
        """
        if token_id in self.children:
            return self.children[token_id]

        child = AdvTreeNode(
            token_id=token_id,
            parent=self,
            depth=self.depth + 1,
        )
        self.children[token_id] = child
        return child

    def update_value(self, value: float, loss: float = None, iteration: int = -1,
                     evaluation_round: int = None):
        """更新节点价值和 loss"""
        self.V = value
        if loss is not None:
            self.best_loss = min(self.best_loss, loss)
            if evaluation_round is not None:
                self.current_loss = loss
                self.evaluation_round = evaluation_round
        self.last_eval_iteration = iteration

    def get_best_child_by_value(self) -> Optional['AdvTreeNode']:
        """返回价值最高的子节点"""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: c.V)

    def get_best_child_by_loss(self) -> Optional['AdvTreeNode']:
        """返回 loss 最低的子节点"""
        if not self.children:
            return None
        return min(self.children.values(), key=lambda c: c.best_loss)

    # ================================================================
    #  收集候选后缀
    # ================================================================

    def collect_complete_suffixes(self, tokenizer, top_k: int = 5) -> List[SuffixCandidate]:
        """
        收集树中所有完整的对抗后缀（深度 = suffix_len 的节点）。

        由于不同节点的 V/loss 是在不同批次上计算的，这里只是收集候选，
        真正的排序需要在相同批次上重新评估。

        Args:
            tokenizer: 用于解码 token 的分词器
            top_k: 返回的候选数量上限

        Returns:
            candidates: SuffixCandidate 列表，按历史 loss 排序（仅供参考）
        """
        candidates = []
        self._collect_complete_suffixes_recursive(candidates, tokenizer)

        # 按历史 loss 排序（注意：这只是启发式排序，不保证准确性）
        candidates.sort(key=lambda c: c.historical_loss)

        return candidates[:top_k]

    def _collect_complete_suffixes_recursive(self, candidates: List[SuffixCandidate],
                                              tokenizer) -> None:
        """递归收集完整后缀"""
        if self.is_complete_suffix:
            suffix_tokens = self.build_full_suffix()
            candidates.append(SuffixCandidate(
                suffix_tokens=suffix_tokens,
                suffix_str=tokenizer.decode(suffix_tokens, skip_special_tokens=True),
                depth=self.depth,
                historical_loss=self.best_loss,
                historical_V=self.V,
                path_tokens=self.get_path_tokens()
            ))

        for child in self.children.values():
            child._collect_complete_suffixes_recursive(candidates, tokenizer)

    def collect_promising_nodes(self, tokenizer, top_k: int = 10,
                                 min_depth: int = 10) -> List[SuffixCandidate]:
        """
        收集树中有潜力的节点（不一定是完整后缀）。

        选择标准：
        1. 深度 >= min_depth（有足够的 token 被优化）
        2. 访问次数较多（被 MCTS 认为有价值）
        3. 历史 loss 较低

        Args:
            tokenizer: 分词器
            top_k: 返回数量
            min_depth: 最小深度要求

        Returns:
            candidates: SuffixCandidate 列表
        """
        candidates = []
        self._collect_promising_nodes_recursive(candidates, tokenizer, min_depth)

        # 综合排序：考虑深度、访问次数和 loss
        # 深度越大越好，访问次数越多越好，loss 越低越好
        candidates.sort(key=lambda c: (
            -c.depth,  # 优先深度大的
            c.historical_loss  # 然后 loss 低的
        ))

        return candidates[:top_k]

    def _collect_promising_nodes_recursive(self, candidates: List[SuffixCandidate],
                                            tokenizer, min_depth: int) -> None:
        """递归收集有潜力的节点"""
        if self.depth >= min_depth and self.numVisits > 0:
            suffix_tokens = self.build_full_suffix()
            candidates.append(SuffixCandidate(
                suffix_tokens=suffix_tokens,
                suffix_str=tokenizer.decode(suffix_tokens, skip_special_tokens=True),
                depth=self.depth,
                historical_loss=self.best_loss,
                historical_V=self.V,
                path_tokens=self.get_path_tokens()
            ))

        for child in self.children.values():
            child._collect_promising_nodes_recursive(candidates, tokenizer, min_depth)

    # ================================================================
    #  递归查询（保留，但标注跨批次不可比）
    # ================================================================

    def getBestV(self) -> Tuple['AdvTreeNode', float]:
        """
        递归查找子树中价值最高的节点。

        注意：不同节点的 V 是在不同批次上计算的，此方法返回的结果
        仅供参考，真正比较需要在相同批次上重新评估。
        """
        best_node = self
        best_V = self.V

        for child in self.children.values():
            sub_node, sub_V = child.getBestV()
            if sub_V > best_V:
                best_V = sub_V
                best_node = sub_node

        return best_node, best_V

    def get_best_loss_node(self) -> Tuple['AdvTreeNode', float]:
        """
        递归查找子树中 loss 最低的节点。

        注意：不同节点的 loss 是在不同批次上计算的，此方法返回的结果
        仅供参考，真正比较需要在相同批次上重新评估。
        """
        best_node = self
        best_loss = self.best_loss

        for child in self.children.values():
            sub_node, sub_loss = child.get_best_loss_node()
            if sub_loss < best_loss:
                best_loss = sub_loss
                best_node = sub_node

        return best_node, best_loss

    def get_adv_suffix_string(self, tokenizer) -> str:
        """将当前节点的完整后缀 token 序列解码为字符串"""
        suffix_tokens = self.build_full_suffix()
        return tokenizer.decode(suffix_tokens, skip_special_tokens=True)

    def get_tree_stats(self) -> dict:
        """获取以当前节点为根的子树统计信息"""
        total_nodes = 1
        total_visits = self.numVisits
        max_depth = self.depth
        min_loss = self.best_loss
        complete_suffixes = 1 if self.is_complete_suffix else 0

        for child in self.children.values():
            stats = child.get_tree_stats()
            total_nodes += stats['total_nodes']
            total_visits += stats['total_visits']
            max_depth = max(max_depth, stats['max_depth'])
            min_loss = min(min_loss, stats['min_loss'])
            complete_suffixes += stats['complete_suffixes']

        return {
            'total_nodes': total_nodes,
            'total_visits': total_visits,
            'max_depth': max_depth,
            'min_loss': min_loss,
            'complete_suffixes': complete_suffixes
        }

    def collect_best_candidates(self, tokenizer, top_k: int = 5,
                                evaluation_round: int = None,
                                exclude_evaluation_round: int = None) -> List[SuffixCandidate]:
        """
        多样性感知的候选收集。

        指定 evaluation_round 时，只收集当前图像上本轮实际评估过的节点，
        并按 current_loss 排序；历史 best_loss 不参与活跃候选初筛。
        """
        # Phase 1: 收集所有已评估节点（包括非叶节点，代表不同深度的部分路径）
        evaluated_nodes = []
        self._collect_evaluated_nodes(
            evaluated_nodes,
            evaluation_round,
            exclude_evaluation_round,
        )

        # Phase 2: 活跃搜索按本轮真实 loss 排序；兼容调用仍可使用历史 loss。
        evaluated_nodes.sort(key=lambda n: (
            n.current_loss if evaluation_round is not None else n.best_loss,
            -n.depth,
        ))

        # Phase 3: 多样性去重 —— 优先选择 early-position 不同的候选
        selected = []
        seen_suffixes = set()
        # 追踪已占用的 early positions (depth -> set of token_ids)
        early_pos_used = {}

        for node in evaluated_nodes:
            if len(selected) >= top_k:
                break

            suffix = node.build_full_suffix()
            suffix_key = tuple(suffix.tolist())
            if suffix_key in seen_suffixes:
                continue

            # 多样性检查: 如果候选的前几个位置与已选候选完全相同，跳过
            path_tokens = node.get_path_tokens()
            path_positions = node.get_path_positions()
            is_diverse = True
            if len(selected) > 0 and len(path_tokens) > 0:
                # 检查本轮循环起点是否已有足够多样性
                first_pos = path_positions[0]
                first_tok = path_tokens[0]
                if first_pos not in early_pos_used:
                    early_pos_used[first_pos] = set()
                if first_tok in early_pos_used[first_pos] and len(early_pos_used[first_pos]) >= 2:
                    # depth=1 已有 2+ 种不同 token，允许重复
                    pass
                elif first_tok in early_pos_used[first_pos]:
                    # depth=1 还缺乏多样性: 只有当这个候选 loss 显著更优时才考虑
                    if selected:
                        best_loss = selected[0].current_loss
                        node_loss = (
                            node.current_loss
                            if evaluation_round is not None else node.best_loss
                        )
                        if node_loss >= best_loss * 0.95:
                            is_diverse = False

            if not is_diverse:
                continue

            # 记录 early positions
            for pos, tok_id in zip(path_positions, path_tokens):
                if pos not in early_pos_used:
                    early_pos_used[pos] = set()
                early_pos_used[pos].add(tok_id)

            seen_suffixes.add(suffix_key)
            selected.append(SuffixCandidate(
                suffix_tokens=suffix,
                suffix_str=tokenizer.decode(suffix, skip_special_tokens=True),
                depth=node.depth,
                historical_loss=node.best_loss,
                historical_V=node.V,
                path_tokens=path_tokens,
                node=node,
                current_loss=(
                    node.current_loss
                    if evaluation_round is not None else node.best_loss
                ),
                evaluation_round=(
                    node.evaluation_round
                    if evaluation_round is not None else -1
                ),
                recall_source=(
                    'current' if evaluation_round is not None else 'historical'
                ),
            ))

        return selected

    def collect_deep_candidates(self, tokenizer, top_k: int,
                                min_depth: int,
                                evaluation_round: int = None,
                                exclude_evaluation_round: int = None) -> List[SuffixCandidate]:
        """收集当前轮达到接受深度门槛的最优候选。

        不使用访问次数作为过滤条件，因为 rollout 中已评估的深层节点可能
        尚未独立 backprop。返回值与 collect_best_candidates 合并，避免浅层
        低 loss 节点占满候选池。
        """
        if top_k <= 0:
            return []

        evaluated_nodes = []
        self._collect_evaluated_nodes(
            evaluated_nodes,
            evaluation_round,
            exclude_evaluation_round,
        )
        evaluated_nodes = [node for node in evaluated_nodes
                           if node.depth >= min_depth]
        evaluated_nodes.sort(key=lambda node: (
            node.current_loss if evaluation_round is not None else node.best_loss,
            -node.depth,
        ))

        selected = []
        seen_suffixes = set()
        for node in evaluated_nodes:
            if len(selected) >= top_k:
                break
            suffix = node.build_full_suffix()
            suffix_key = tuple(suffix.tolist())
            if suffix_key in seen_suffixes:
                continue
            seen_suffixes.add(suffix_key)
            selected.append(SuffixCandidate(
                suffix_tokens=suffix,
                suffix_str=tokenizer.decode(suffix, skip_special_tokens=True),
                depth=node.depth,
                historical_loss=node.best_loss,
                historical_V=node.V,
                path_tokens=node.get_path_tokens(),
                node=node,
                current_loss=(
                    node.current_loss
                    if evaluation_round is not None else node.best_loss
                ),
                evaluation_round=(
                    node.evaluation_round
                    if evaluation_round is not None else -1
                ),
                recall_source=(
                    'current_deep'
                    if evaluation_round is not None else 'historical_deep'
                ),
            ))

        return selected

    def _collect_evaluated_nodes(self, nodes: list,
                                 evaluation_round: int = None,
                                 exclude_evaluation_round: int = None) -> None:
        """递归收集已评估节点，可选择指定或排除某一评估轮次。"""
        if evaluation_round is None:
            is_evaluated = self.best_loss < float('inf')
        else:
            is_evaluated = (
                self.evaluation_round == evaluation_round
                and self.current_loss < float('inf')
            )
        if (exclude_evaluation_round is not None and
                self.evaluation_round == exclude_evaluation_round):
            is_evaluated = False
        if not self.is_root and is_evaluated:
            nodes.append(self)
        for child in self.children.values():
            child._collect_evaluated_nodes(
                nodes,
                evaluation_round,
                exclude_evaluation_round,
            )
           
    # def decay_tree_values(self, decay_factor: float = 0.5):
    #     self.V *= decay_factor
    #     self.unexpanded_candidates = None
    #     self.all_candidates_pool = None  # 梯度完全失效，下次重算
    #     self.pool_index = 0
        
    #     for child in self.children.values():
    #         child.decay_tree_values(decay_factor)
    
    def rescale_tree_values(self, scale: float, pw_alpha: float = 0.5):
        """
        等比缩放子树 V 值并重置搜索状态。

        numVisits 重置为 len(children)^2，使得
        pw_limit = ceil(sqrt(k^2)) = k = len(children)，虚拟动作关闭。
        第 1 次访问：纯走已有子节点 → rollout 用真实梯度更新 V 值。
        第 2 次访问：numVisits+1 → pw_limit = k+1 > k → 开启 1 次扩展。
        配合剪枝（10 条路径），前 ~10 次搜索校准 V，后 ~10 次扩展。

        重置 has_been_rolled_out：持续为 True 会导致 search_continue 中
        旧子节点被选中后跳过 rollout，迭代只扩展不增深度。
        """
        self.V *= scale
        n_children = len(self.children)
        if n_children > 0:
            # k^2: pw_limit = ceil(sqrt(k^2)) = k = n_children
            # 第1次访问: 虚拟动作关闭，纯走已有子节点 → rollout 更新 V
            # 第2次访问: numVisits+1 → pw_limit = k+1 > k → 开启扩展
            self.numVisits = max(1, n_children * n_children)
        else:
            self.numVisits = 1
        self.has_been_rolled_out = False
        self.unexpanded_candidates = None
        self.all_candidates_pool = None
        self.pool_index = 0
        for child in self.children.values():
            child.rescale_tree_values(scale, pw_alpha)

    def prune_to_paths(self, keep_nodes: list):
        """
        修剪树：只保留 keep_nodes 中每个节点到根的路径上的节点，
        其余分支全部删除。

        Args:
            keep_nodes: 希望保留的叶节点列表
        Returns:
            保留的节点总数
        """
        # 收集所有需要保留的节点（包括到根的全部祖先）
        keep_set = set()
        for node in keep_nodes:
            current = node
            while current is not None:
                if id(current) in keep_set:
                    break  # 已经标记过，祖先也已标记
                keep_set.add(id(current))
                current = current.parent

        # 从根开始递归删除不在 keep_set 中的分支
        count = self._prune_recursive(keep_set)
        return count

    def _prune_recursive(self, keep_set: set) -> int:
        """递归修剪子树，返回保留的节点数"""
        to_remove = [tok for tok, child in self.children.items()
                     if id(child) not in keep_set]
        for tok in to_remove:
            del self.children[tok]

        count = 1  # 自身
        for child in self.children.values():
            count += child._prune_recursive(keep_set)
        return count

    def __str__(self):
        if self.is_root:
            return (f"AdvTreeNode(ROOT, depth={self.depth}, visits={self.numVisits}, "
                    f"V={self.V:.4f}, loss={self.best_loss:.4f}, "
                    f"children={len(self.children)})")
        return (f"AdvTreeNode(pos={self.position}, tok={self.token_id}, "
                f"depth={self.depth}, visits={self.numVisits}, "
                f"V={self.V:.4f}, loss={self.best_loss:.4f}, "
                f"children={len(self.children)})")

    def __repr__(self):
        return self.__str__()


class AdvMCTSConfig:
    """MCTS 对抗搜索的配置类"""

    def __init__(
        self,
        # MCTS 核心参数
        exploration_constant: float = 1.5,  # 提高探索权重 (0.7 → 1.5)
        iteration_limit: int = 50,

        # 扩展参数
        branch: int = 16,              # 每个位置的候选 token 数（树宽度）
        topk: int = 256,               # 梯度 Top-k 候选 token 数
        max_branch: int = 32,          # 候选池上限（存储的梯度候选数）

        # 渐进加宽参数
        pw_alpha: float = 0.7,         # 渐进加宽指数: ceil(N^0.7), N=4→3, N=10→5
        
        # Rollout 参数
        roll_forward_steps: int = 5,   # 贪婪前瞻的位置数
        alpha: float = 0.5,            # 加权平均的平滑系数

        # 奖励映射
        reward_beta: float = 1.0,

        # 反向传播策略
        backup_strategy: str = 'mean',  # 'max' or 'weighted_avg'

        # 候选返回参数
        num_candidates_return: int = 10,  # 返回的候选后缀数量

        # Re-root 后的位置覆盖与接受门控
        min_accept_depth: int = 0,  # 共享搜索器默认不启用接受门控
        deep_accept_improvement: float = 0.0,
        shallow_accept_improvement: float = 0.05,
        position_stride: int = 5,
        full_rollout_after_reroot: bool = False,

        # 历史节点只用于召回，候选会在当前图像上重新评估
        historical_candidate_quota: int = 0,

        # 修剪参数
        prune_keep_paths: int = 10,       # 每轮搜索后保留的最优路径数

        # ============================================================
        # Embedding 空间采样参数（替代 GCG 梯度）
        # ============================================================
        use_embedding_sampling: bool = False,    # 是否启用 embedding 空间采样
        embedding_num_samples: int = 64,          # expand/greedy 中采样的候选数
        embedding_sample_temperature: float = 1.0, # embedding 相似度 softmax 温度
        embedding_uniform_ratio: float = 0.3,     # 均匀采样的比例（0=纯embedding相似度）
        greedy_eval_candidates: int = 3,          # greedy_rollout 首轮评估的候选数
        greedy_eval_candidates_max: int = 8,      # 首轮无明显改善时的最大评估数
        greedy_expand_threshold: float = 0.01,    # 相对当前 token 的最小 loss 改善

        # ============================================================
        # LM 先验参数（Bigram 频率表）
        # ============================================================
        lm_prior_weight: float = 0.0,               # LM prior 混合权重 [0, 1], 0=禁用
        bigram_table_path: str = "",                 # 预计算 bigram 表 JSON 路径

        # 后缀长度（从输入推断，此处作为默认值）
        suffix_len: int = 20,

        # UCB 参数
        INF: float = 1.0,

        # 日志
        verbose: bool = True,

    ):
        self.exploration_constant = exploration_constant
        self.iteration_limit = iteration_limit
 
        self.branch = branch
        self.topk = topk
        self.max_branch = max_branch
        self.pw_alpha = pw_alpha

        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha

        self.reward_beta = reward_beta
        self.backup_strategy = backup_strategy

        self.num_candidates_return = num_candidates_return
        self.min_accept_depth = min_accept_depth
        self.deep_accept_improvement = deep_accept_improvement
        self.shallow_accept_improvement = shallow_accept_improvement
        self.position_stride = position_stride
        self.full_rollout_after_reroot = full_rollout_after_reroot
        self.historical_candidate_quota = max(0, historical_candidate_quota)
        self.prune_keep_paths = prune_keep_paths

        self.use_embedding_sampling = use_embedding_sampling
        self.embedding_num_samples = embedding_num_samples
        self.embedding_sample_temperature = embedding_sample_temperature
        self.embedding_uniform_ratio = embedding_uniform_ratio
        self.greedy_eval_candidates = greedy_eval_candidates
        self.greedy_eval_candidates_max = greedy_eval_candidates_max
        self.greedy_expand_threshold = greedy_expand_threshold

        self.lm_prior_weight = lm_prior_weight
        self.bigram_table_path = bigram_table_path

        self.suffix_len = suffix_len

        self.INF = INF
        self.verbose = verbose


    def loss_to_reward(self, loss: float) -> float:
        """将交叉熵 loss 映射为奖励值 [0, 1]"""
        return math.exp(-self.reward_beta * loss)

    def normalized_min_accept_depth(self, suffix_len: int) -> int:
        """将接受深度限制在当前后缀长度内。"""
        if suffix_len <= 0:
            raise ValueError("suffix_len must be positive")
        return min(max(0, self.min_accept_depth), suffix_len)

    def should_accept_candidate(self, depth: int, improvement: float,
                                suffix_len: int) -> bool:
        """深浅候选都必须达到各自的最小改善。"""
        if improvement <= 0:
            return False
        return improvement >= self.required_accept_improvement(depth, suffix_len)

    def required_accept_improvement(self, depth: int, suffix_len: int) -> float:
        """返回候选在当前深度下需要达到的 loss 改善。"""
        min_depth = self.normalized_min_accept_depth(suffix_len)
        deep_threshold = max(0.0, self.deep_accept_improvement)
        if depth >= min_depth:
            return deep_threshold
        return max(deep_threshold, self.shallow_accept_improvement)

    def passes_validation(self, depth: int, proposal_improvement: float,
                          validation_improvement: float,
                          suffix_len: int) -> bool:
        """验证 batch 不得退化，且两个 batch 的平均改善需达标。"""
        if validation_improvement <= 0:
            return False
        mean_improvement = (
            proposal_improvement + validation_improvement
        ) / 2.0
        return mean_improvement >= self.required_accept_improvement(
            depth, suffix_len
        )

    def next_start_offset(self, current_offset: int, suffix_len: int) -> int:
        """在接受新后缀后计算下一棵树的循环起点。"""
        if suffix_len <= 0:
            raise ValueError("suffix_len must be positive")
        return (current_offset + self.position_stride) % suffix_len
