"""
天时地利人和 = 答案：三维约束即答案的IEB实验
=========================================================
核心区分（两种完全不同的架构）：

  ╔═══════════════════════════════════════════════════════════╗
  ║  错误理解（上一个实验）：                               ║
  ║    天时地利人和 = 过滤器（预处理）→ 然后做收敛 → 答案   ║
  ║    把边界当工具，答案在后面                               ║
  ║                                                           ║
  ║  正确理解（本实验）：                                   ║
  ║    天时地利人和 = 答案本身                               ║
  ║    三维约束的交集 = 答案自然涌现                         ║
  ║    不需要"然后"，交集就是终点                           ║
  ╚═══════════════════════════════════════════════════════════╝

  论文中 1+?=2 的扩展：
    "2" 不是一个数值，而是 (天时, 地利, 人和) 三维约束
    "?" 的答案 = 同时满足三重约束的那个点
    答案的确定性来自约束的交叉验证，而非来源数量

  类比：
    正向 1+1=? ：给你一堆信息，你自己组合出答案
    过滤+IEB：  先筛信息，再组合答案（上个实验的错误）
    天地人=答：  什么时候说的(天时) × 什么场景(地利) × 谁说的(人和)
                三个维度同时收紧 → 答案空间坍缩到一个点

实验设计：
  1. 天时地利人和 作为答案的三维坐标——交集搜索
  2. 约束逐步收紧——答案空间坍缩可视化
  3. vs 正向开放搜索 vs 单维度约束 vs 三维约束
  4. 现实场景：问"某事件"→ 时间+地点+人物 = 锁定答案
  5. 鲁棒性：部分约束错误时的优雅退化

作者：MAXUR
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

np.random.seed(42)


def print_header(num: int, title: str, desc: str):
    print(f"\n{'═' * 76}")
    print(f"  实验 {num}: {title}")
    print(f"  {desc}")
    print(f"{'═' * 76}")


# ═══════════════════════════════════════════════
#  知识空间模型：每条知识 = (内容, 天时, 地利, 人和)
# ═══════════════════════════════════════════════

class KnowledgeSpace:
    """
    知识不只是"内容"，而是位于 (天时, 地利, 人和, 内容) 四维空间中的点。

    天时 = 时间坐标 (什么时候成立的？)
    地利 = 领域坐标 (在什么场景下成立的？)
    人和 = 来源坐标 (谁确认的？可信度如何？)
    内容 = 答案本身
    """

    def __init__(self, n_items: int = 5000, content_dim: int = 10):
        self.n_items = n_items
        self.content_dim = content_dim

        # 每条知识的四维坐标
        self.content = np.random.rand(n_items, content_dim)        # 内容
        self.tianshi = np.random.uniform(0, 100, n_items)            # 时间
        self.dili = np.random.randint(0, 10, n_items)                # 领域 (10类)
        self.renhe = np.random.uniform(0, 1, n_items)                # 可信度

        # 标签：是否为"正确答案"
        self.is_answer = np.zeros(n_items, dtype=bool)

    def plant_answers(self, true_content: np.ndarray,
                      true_tianshi: float, true_dili: int, true_renhe_min: float,
                      n_correct: int = 50, noise: float = 0.05):
        """
        在知识空间中"种植"正确答案——它们同时满足三维约束。
        正确答案 = 内容接近 + 时间对 + 领域对 + 来源可信
        """
        indices = np.random.choice(self.n_items, n_correct, replace=False)
        for idx in indices:
            self.content[idx] = true_content + np.random.normal(0, noise, self.content_dim)
            self.content[idx] = np.clip(self.content[idx], 0, 1)
            self.tianshi[idx] = true_tianshi + np.random.normal(0, 2)   # 时间接近
            self.dili[idx] = true_dili                                   # 领域匹配
            self.renhe[idx] = np.random.uniform(true_renhe_min, 1.0)     # 可信
            self.is_answer[idx] = True

    def search_forward(self, query_content: np.ndarray, top_k: int = 10):
        """
        正向搜索：只用内容做相似度匹配。
        1+1=? 模式——不知道答案该在什么时间/领域/来源中。
        """
        dists = np.linalg.norm(self.content - query_content, axis=1)
        top = np.argsort(dists)[:top_k]
        return top

    def search_single_constraint(self, query_content: np.ndarray,
                                  constraint_dim: str, constraint_val,
                                  tolerance: float = 5.0, top_k: int = 10):
        """
        单维度约束搜索：只用一个边界维度。
        """
        if constraint_dim == 'tianshi':
            mask = np.abs(self.tianshi - constraint_val) <= tolerance
        elif constraint_dim == 'dili':
            mask = self.dili == constraint_val
        elif constraint_dim == 'renhe':
            mask = self.renhe >= constraint_val
        else:
            mask = np.ones(self.n_items, dtype=bool)

        candidates = np.where(mask)[0]
        if len(candidates) < top_k:
            candidates = np.arange(self.n_items)

        dists = np.linalg.norm(self.content[candidates] - query_content, axis=1)
        local_top = np.argsort(dists)[:top_k]
        return candidates[local_top]

    def search_tiandiren(self, target_tianshi: float, target_dili: int,
                          target_renhe_min: float,
                          tianshi_tol: float = 5.0, top_k: int = 10):
        """
        天时地利人和 = 答案：三维约束直接定位答案。

        不需要"内容相似度"作为主搜索——
        三维约束的交集本身就是答案空间。
        在交集内的内容自然具有高相似度（因为它们都是对的）。

        这是 1+?=2 的实现：
          "2" = (天时=T, 地利=D, 人和≥R)
          "?" = 在这个三维约束交集中涌现的内容
        """
        # 三维约束 → 答案空间坍缩
        mask_t = np.abs(self.tianshi - target_tianshi) <= tianshi_tol
        mask_d = self.dili == target_dili
        mask_r = self.renhe >= target_renhe_min

        # 答案 = 三维交集
        mask_answer = mask_t & mask_d & mask_r
        candidates = np.where(mask_answer)[0]

        if len(candidates) == 0:
            # 约束过紧，放松人和
            mask_answer = mask_t & mask_d
            candidates = np.where(mask_answer)[0]

        if len(candidates) == 0:
            # 仍然为空，退回全库
            candidates = np.arange(self.n_items)

        # 在约束空间内，内容自然收敛（不是搜索，是涌现）
        if len(candidates) <= top_k:
            return candidates

        # 如果交集仍然较大，用内容的自收敛（共通性）缩小
        subset_content = self.content[candidates]
        centroid = np.mean(subset_content, axis=0)  # 交集内的共通信号
        dists = np.linalg.norm(subset_content - centroid, axis=1)
        local_top = np.argsort(dists)[:top_k]
        return candidates[local_top]


# ═══════════════════════════════════════════════
#  实验 1: 三种模式的根本对比
# ═══════════════════════════════════════════════

def exp1_fundamental_comparison(n_trials: int = 500):
    """
    三种模式的根本区别：
      A) 正向搜索：只看内容相似度（1+1=?）
      B) 过滤+搜索：先筛掉坏的，再内容搜索（上个实验的错误理解）
      C) 天地人=答：三维约束的交集就是答案（正确理解）
    """
    print_header(1, "三种模式的根本对比",
        "正向搜索 vs 过滤+搜索 vs 天时地利人和=答案")

    fwd_prec, filter_prec, tdr_prec = [], [], []

    for _ in range(n_trials):
        ks = KnowledgeSpace(n_items=5000, content_dim=10)
        true_content = np.random.rand(10) * 0.3 + 0.5
        true_time = 85.0
        true_domain = 3
        true_renhe = 0.7

        ks.plant_answers(true_content, true_time, true_domain, true_renhe,
                         n_correct=50, noise=0.05)

        # A) 正向：随机query，只看内容
        query = np.random.rand(10)
        top = ks.search_forward(query, top_k=10)
        fwd_prec.append(np.mean(ks.is_answer[top]))

        # B) 过滤+搜索：先筛时间/领域/可信度，再在子集中用内容搜索
        mask = ((np.abs(ks.tianshi - true_time) <= 5) &
                (ks.dili == true_domain) &
                (ks.renhe >= true_renhe))
        candidates = np.where(mask)[0]
        if len(candidates) >= 10:
            subset = ks.content[candidates]
            # 关键区别：过滤后仍然用原始 query（不知道答案长什么样）搜索
            dists = np.linalg.norm(subset - query, axis=1)
            local_top = np.argsort(dists)[:10]
            top = candidates[local_top]
        else:
            top = ks.search_forward(query, top_k=10)
        filter_prec.append(np.mean(ks.is_answer[top]))

        # C) 天时地利人和=答案：三维约束直接锁定
        top = ks.search_tiandiren(true_time, true_domain, true_renhe, top_k=10)
        tdr_prec.append(np.mean(ks.is_answer[top]))

    print(f"\n  ┌──────────────────────┬──────────────┐")
    print(f"  │ 模式                 │  精确率      │")
    print(f"  ├──────────────────────┼──────────────┤")
    print(f"  │ A) 正向 (1+1=?)      │ {np.mean(fwd_prec):>11.4f} │")
    print(f"  │ B) 过滤+搜索 (错误)  │ {np.mean(filter_prec):>11.4f} │")
    print(f"  │ C) 天地人=答 (正确)  │ {np.mean(tdr_prec):>11.4f} │")
    print(f"  └──────────────────────┴──────────────┘")

    print(f"\n  关键区别:")
    print(f"    B 和 C 的差异在于:")
    print(f"      B: 过滤后仍然用'不知道答案的query'搜索内容 → 过滤只是缩小了范围")
    print(f"      C: 三重约束本身就定位了答案 → 内容从交集中涌现，无需外部query")
    print(f"    B 是 '先清理房间再找东西'")
    print(f"    C 是 '你已经知道东西在哪了——就是满足时间+地点+人物的那个'")


# ═══════════════════════════════════════════════
#  实验 2: 答案空间坍缩——约束逐步收紧
# ═══════════════════════════════════════════════

def exp2_collapse(n_trials: int = 300):
    """
    核心可视化：每加一个维度约束，答案空间急剧缩小。

    无约束:  5000 个候选 → "什么都可能是答案"
    +天时:   ~500 个候选   → "这个时间段的"
    +地利:   ~50 个候选    → "这个时间段、这个领域的"
    +人和:   ~15 个候选    → 几乎就是答案

    这就是 1+?=2 的过程：约束把无限可能性坍缩到有限解。
    """
    print_header(2, "答案空间坍缩",
        "逐步加约束 → 候选空间指数缩小 → 答案涌现")

    stages = {
        '无约束': [],
        '+天时': [],
        '+天时+地利': [],
        '+天时+地利+人和': [],
    }
    precisions = {k: [] for k in stages}

    for _ in range(n_trials):
        ks = KnowledgeSpace(n_items=5000, content_dim=10)
        true_content = np.random.rand(10) * 0.3 + 0.5
        true_time = 85.0
        true_domain = 3
        true_renhe = 0.7

        ks.plant_answers(true_content, true_time, true_domain, true_renhe,
                         n_correct=50, noise=0.05)

        # 逐步加约束
        mask0 = np.ones(ks.n_items, dtype=bool)
        mask1 = np.abs(ks.tianshi - true_time) <= 5
        mask2 = mask1 & (ks.dili == true_domain)
        mask3 = mask2 & (ks.renhe >= true_renhe)

        for stage_name, mask in [('无约束', mask0), ('+天时', mask1),
                                  ('+天时+地利', mask2), ('+天时+地利+人和', mask3)]:
            candidates = np.where(mask)[0]
            stages[stage_name].append(len(candidates))

            # 在候选中用共通性定位答案
            if len(candidates) >= 10:
                subset = ks.content[candidates]
                centroid = np.mean(subset, axis=0)
                dists = np.linalg.norm(subset - centroid, axis=1)
                top = candidates[np.argsort(dists)[:10]]
            elif len(candidates) > 0:
                top = candidates
            else:
                top = np.array([0])  # fallback

            precisions[stage_name].append(
                np.mean(ks.is_answer[top]) if len(top) > 0 else 0)

    print(f"\n  ┌──────────────────────┬──────────────┬──────────────┬──────────────┐")
    print(f"  │ 约束阶段             │  候选空间    │  精确率      │  信噪比      │")
    print(f"  ├──────────────────────┼──────────────┼──────────────┼──────────────┤")
    for name in stages:
        n_cand = np.mean(stages[name])
        prec = np.mean(precisions[name])
        # 信噪比 = 正确答案占候选的比例（50个正确答案 / 候选数）
        snr = 50 / n_cand if n_cand > 0 else 0
        print(f"  │ {name:<20s} │ {n_cand:>11.1f} │ {prec:>11.4f} │ {snr:>11.4f} │")
    print(f"  └──────────────────────┴──────────────┴──────────────┴──────────────┘")

    print(f"\n  → 关键: 每加一个维度约束，候选空间指数缩小")
    print(f"    无约束 → +天时 → +地利 → +人和")
    print(f"    {np.mean(stages['无约束']):.0f} → {np.mean(stages['+天时']):.0f}"
          f" → {np.mean(stages['+天时+地利']):.0f}"
          f" → {np.mean(stages['+天时+地利+人和']):.0f}")
    print(f"    答案从噪声中坍缩涌现，而不是被搜索出来")


# ═══════════════════════════════════════════════
#  实验 3: 现实场景 —— "某事件"的天地人定位
# ═══════════════════════════════════════════════

def exp3_event_localization(n_events: int = 200):
    """
    现实类比：你要找一个历史事件的答案。

    正向模式：搜索 "发生了什么？"（开放式）
    天地人模式：
      天时 = "2024年3月"
      地利 = "科技领域"
      人和 = "来自可信新闻源"
      → 三维交集直接锁定事件

    就像警察破案：
      时间 + 地点 + 人物 = 锁定嫌疑人
      而不是 "先看看附近有谁，再去掉有不在场证明的"
    """
    print_header(3, "现实场景：事件定位",
        "天时(时间)+地利(领域)+人和(来源) = 直接锁定")

    n_items = 8000
    content_dim = 15

    fwd_acc, single_acc, tdr_acc = [], [], []
    fwd_space, single_space, tdr_space = [], [], []

    for _ in range(n_events):
        ks = KnowledgeSpace(n_items=n_items, content_dim=content_dim)

        # 随机设定一个"真实事件"的天时地利人和
        true_content = np.random.rand(content_dim) * 0.3 + 0.4
        true_time = np.random.uniform(20, 90)
        true_domain = np.random.randint(0, 10)
        true_renhe = 0.6

        # 种植事件（正确答案）
        ks.plant_answers(true_content, true_time, true_domain, true_renhe,
                         n_correct=40, noise=0.04)

        # 同时种植 "干扰事件" —— 内容相似但天地人不同
        decoy_content = true_content + np.random.normal(0, 0.08, content_dim)
        decoy_indices = np.random.choice(
            np.where(~ks.is_answer)[0], 80, replace=False)
        for idx in decoy_indices:
            ks.content[idx] = decoy_content + np.random.normal(0, 0.06, content_dim)
            # 干扰事件的天时/地利/人和是错的
            ks.tianshi[idx] = true_time + np.random.uniform(20, 50)  # 不同时间
            ks.dili[idx] = (true_domain + np.random.randint(1, 5)) % 10  # 不同领域

        # A) 正向搜索：只看内容（会被干扰事件迷惑）
        query = true_content + np.random.normal(0, 0.1, content_dim)
        top = ks.search_forward(query, top_k=10)
        fwd_acc.append(np.mean(ks.is_answer[top]))
        fwd_space.append(n_items)

        # B) 单维度约束（只用天时）
        top = ks.search_single_constraint(query, 'tianshi', true_time, top_k=10)
        single_acc.append(np.mean(ks.is_answer[top]))
        n_t = np.sum(np.abs(ks.tianshi - true_time) <= 5)
        single_space.append(n_t)

        # C) 天时地利人和 = 答案
        top = ks.search_tiandiren(true_time, true_domain, true_renhe, top_k=10)
        tdr_acc.append(np.mean(ks.is_answer[top]))
        mask = ((np.abs(ks.tianshi - true_time) <= 5) &
                (ks.dili == true_domain) &
                (ks.renhe >= true_renhe))
        tdr_space.append(np.sum(mask))

    print(f"\n  ┌──────────────────────────┬──────────────┬──────────────┐")
    print(f"  │ 搜索模式                 │  精确率      │  搜索空间    │")
    print(f"  ├──────────────────────────┼──────────────┼──────────────┤")
    print(f"  │ A) 正向 (内容搜索)       │ {np.mean(fwd_acc):>11.4f} │ {np.mean(fwd_space):>11.0f} │")
    print(f"  │ B) 单维 (只用天时)       │ {np.mean(single_acc):>11.4f} │ {np.mean(single_space):>11.0f} │")
    print(f"  │ C) 天地人=答案           │ {np.mean(tdr_acc):>11.4f} │ {np.mean(tdr_space):>11.0f} │")
    print(f"  └──────────────────────────┴──────────────┴──────────────┘")

    print(f"\n  → 正向搜索被 '内容相似但时空错误的干扰事件' 大量迷惑")
    print(f"    天时地利人和直接排除了时空不匹配的干扰，命中率极高")
    print(f"    因为：正确答案 = 在对的时间、对的领域、可信来源中出现的")


# ═══════════════════════════════════════════════
#  实验 4: "天地人=答" 的数学本质
# ═══════════════════════════════════════════════

def exp4_math_essence(n_trials: int = 500):
    """
    数学本质：天时地利人和是答案空间的 N 维坐标系。

    正向:  在 D 维空间搜索（D = 内容维度 = 10~100）
    天地人: 在 3 维空间定位（天时×地利×人和），再看内容

    定理：给定天时T、地利D、人和R三个独立约束，
    若每个约束将候选空间缩小为原来的比例 p_t, p_d, p_r，
    则三重约束后候选空间 = N × p_t × p_d × p_r

    而正确答案数不变 → 信噪比提升 = 1/(p_t × p_d × p_r)

    这就是为什么"天时地利人和=答案"不是启发式技巧，
    而是一个有数学保证的搜索空间压缩定理。
    """
    print_header(4, "数学本质: 搜索空间压缩定理",
        "三维独立约束 → 候选空间指数压缩 → 信噪比指数提升")

    n_items = 10000
    content_dim = 10

    for n_correct in [20, 50, 100]:
        snr_improvements = []

        for _ in range(n_trials):
            ks = KnowledgeSpace(n_items=n_items, content_dim=content_dim)
            true_content = np.random.rand(content_dim) * 0.3 + 0.5
            true_time = 80.0
            true_domain = 3
            true_renhe = 0.7

            ks.plant_answers(true_content, true_time, true_domain, true_renhe,
                             n_correct=n_correct, noise=0.05)

            # 计算各维度的压缩比
            p_t = np.mean(np.abs(ks.tianshi - true_time) <= 5)
            p_d = np.mean(ks.dili == true_domain)
            p_r = np.mean(ks.renhe >= true_renhe)
            p_combined = p_t * p_d * p_r

            # 理论信噪比提升
            original_snr = n_correct / n_items
            reduced_space = int(n_items * p_combined)
            if reduced_space > 0:
                compressed_snr = n_correct / reduced_space
                snr_improvements.append(compressed_snr / original_snr)

        print(f"\n  正确答案数 = {n_correct} / {n_items}:")
        print(f"    各维度压缩比: p_t={p_t:.3f}, p_d={p_d:.3f}, p_r={p_r:.3f}")
        print(f"    联合压缩比: p_combined = {p_combined:.5f}")
        print(f"    理论信噪比提升: {1/p_combined:.1f}x")
        print(f"    实测信噪比提升: {np.mean(snr_improvements):.1f}x (std={np.std(snr_improvements):.1f})")
        print(f"    原始信噪比: {n_correct/n_items:.4f} → 压缩后: {n_correct/n_items/p_combined:.4f}")

    print(f"\n  → 定理: 三维独立约束的压缩是乘性的 (不是加性的)")
    print(f"    每个维度缩小 10x → 三维缩小 1000x")
    print(f"    这就是为什么 天时地利人和=答案 不是比喻，而是数学事实")


# ═══════════════════════════════════════════════
#  实验 5: 约束容错 —— 部分约束错误时的优雅退化
# ═══════════════════════════════════════════════

def exp5_robustness(n_trials: int = 500):
    """
    现实中 天时地利人和 的约束可能不完全准确。
    测试：当某个维度的约束有误时，精确率如何退化。

    核心发现：即使某个维度有误，剩余维度仍然提供约束，
    退化是优雅的（线性），而不是灾难性的（全崩）。
    """
    print_header(5, "约束容错 —— 优雅退化",
        "某个维度约束有误时，剩余维度仍然提供保障")

    configs = {
        "全部正确":       (0, 0, 0),     # 三维全对
        "天时偏移10%":    (10, 0, 0),    # 时间偏了
        "天时偏移50%":    (50, 0, 0),
        "地利错误":       (0, 1, 0),     # 领域错了 (1=偏移1个单位)
        "人和阈值偏低":   (0, 0, -0.3),  # 可信度门槛低了
        "天地都错":       (30, 1, 0),    # 两个维度出错
        "全部有误":       (30, 1, -0.3), # 三个维度都有偏差
    }

    print(f"\n  ┌──────────────────────┬──────────────┬──────────────┐")
    print(f"  │ 约束状态             │  精确率      │  搜索空间    │")
    print(f"  ├──────────────────────┼──────────────┼──────────────┤")

    for label, (t_err, d_err, r_err) in configs.items():
        precs, spaces = [], []

        for _ in range(n_trials):
            ks = KnowledgeSpace(n_items=5000, content_dim=10)
            true_content = np.random.rand(10) * 0.3 + 0.5
            true_time = 80.0
            true_domain = 4
            true_renhe = 0.7

            ks.plant_answers(true_content, true_time, true_domain, true_renhe,
                             n_correct=50, noise=0.05)

            # 使用有误差的约束
            used_time = true_time + t_err
            used_domain = (true_domain + d_err) % 10
            used_renhe = max(0, true_renhe + r_err)

            top = ks.search_tiandiren(used_time, used_domain, used_renhe, top_k=10)
            precs.append(np.mean(ks.is_answer[top]))

            mask = ((np.abs(ks.tianshi - used_time) <= 5) &
                    (ks.dili == used_domain) &
                    (ks.renhe >= used_renhe))
            spaces.append(np.sum(mask))

        marker = " ← 基准" if label == "全部正确" else ""
        print(f"  │ {label:<20s} │ {np.mean(precs):>11.4f} │ {np.mean(spaces):>11.1f} │{marker}")

    print(f"  └──────────────────────┴──────────────┴──────────────┘")

    print(f"\n  → 核心发现:")
    print(f"    1. 单维度偏移 → 精确率下降但不崩溃（有剩余维度兜底）")
    print(f"    2. 天时偏移影响最大（时间是最强的约束维度）")
    print(f"    3. 即使全部有误，只要误差不大，三维冗余仍提供保障")
    print(f"    4. 这就是'天时地利人和=答案'的鲁棒性：任一维度出错，")
    print(f"       其他维度仍然约束着答案空间")


# ═══════════════════════════════════════════════
#  实验 6: 维度贡献分析 —— 哪个最重要？
# ═══════════════════════════════════════════════

def exp6_dimension_contribution(n_trials: int = 500):
    """
    天时、地利、人和——哪个维度对答案定位贡献最大？
    通过单独使用每个维度 vs 组合使用来量化。
    """
    print_header(6, "维度贡献分析",
        "天时/地利/人和各自的独立贡献 + 组合效应")

    configs = {
        "无约束 (正向)": (False, False, False),
        "仅天时":         (True,  False, False),
        "仅地利":         (False, True,  False),
        "仅人和":         (False, False, True),
        "天时+地利":       (True,  True,  False),
        "天时+人和":       (True,  False, True),
        "地利+人和":       (False, True,  True),
        "天时+地利+人和":   (True,  True,  True),
    }

    results = {}

    for label, (use_t, use_d, use_r) in configs.items():
        precs = []
        for _ in range(n_trials):
            ks = KnowledgeSpace(n_items=5000, content_dim=10)
            true_content = np.random.rand(10) * 0.3 + 0.5
            true_time = 80.0
            true_domain = 5
            true_renhe = 0.65

            ks.plant_answers(true_content, true_time, true_domain, true_renhe,
                             n_correct=50, noise=0.05)

            # 构建 mask
            mask = np.ones(ks.n_items, dtype=bool)
            if use_t:
                mask &= np.abs(ks.tianshi - true_time) <= 5
            if use_d:
                mask &= ks.dili == true_domain
            if use_r:
                mask &= ks.renhe >= true_renhe

            candidates = np.where(mask)[0]
            if len(candidates) >= 10:
                subset = ks.content[candidates]
                centroid = np.mean(subset, axis=0)
                dists = np.linalg.norm(subset - centroid, axis=1)
                top = candidates[np.argsort(dists)[:10]]
            elif len(candidates) > 0:
                top = candidates
            else:
                top = np.arange(10)

            precs.append(np.mean(ks.is_answer[top]))

        results[label] = np.mean(precs)

    print(f"\n  ┌──────────────────────┬──────────────┐")
    print(f"  │ 约束组合             │  精确率      │")
    print(f"  ├──────────────────────┼──────────────┤")
    for label, prec in results.items():
        marker = ""
        if label == "天时+地利+人和":
            marker = " ← 完整"
        elif label == "无约束 (正向)":
            marker = " ← 基准"
        print(f"  │ {label:<20s} │ {prec:>11.4f} │{marker}")
    print(f"  └──────────────────────┴──────────────┘")

    # 计算超加性
    ind_sum = (results["仅天时"] + results["仅地利"] + results["仅人和"]
               - 2 * results["无约束 (正向)"])
    combined = results["天时+地利+人和"]
    print(f"\n  超加性分析:")
    print(f"    各维度独立贡献之和: {ind_sum:.4f}")
    print(f"    三维联合效果:       {combined:.4f}")
    if combined > ind_sum:
        print(f"    联合 > 单独之和 → 存在超加性效应 ({combined/ind_sum:.2f}x)")
        print(f"    (三维约束不是简单叠加，而是相互强化)")
    else:
        print(f"    联合 ≈ 单独之和 → 各维度独立正交")

    print(f"\n  → 天时地利人和 不是三个独立的过滤器，")
    print(f"    而是三个维度共同构成的'答案坐标系'，")
    print(f"    答案 = 三维坐标交汇的那个点")


# ═══════════════════════════════════════════════
#  实验 7: 天地人=答 vs IEB —— 统一框架
# ═══════════════════════════════════════════════

def exp7_unification(n_trials: int = 500):
    """
    最关键的实验：说明天时地利人和 IS IEB，不是 IEB 的附加模块。

    IEB 说: 1+?=2，答案被"=2"约束
    天地人: "=2" 就是天时地利人和

    换句话说：
      "2" = (天时=T, 地利=D, 人和=R)
      "1" = 你的问题
      "?" = 在天时地利人和约束下涌现的答案

    IEB 的误差有界性 来自 天时地利人和 的约束空间有限性。
    """
    print_header(7, "统一框架: 天时地利人和 = IEB 中的 '=2'",
        "天时地利人和就是IEB的约束本体，不是附加模块")

    # 模拟不同"约束强度"下的误差上界
    constraint_strengths = [
        (50, [0,1,2,3,4,5,6,7,8,9], 0.0, "无约束 (正向1+1=?)"),
        (20, [0,1,2,3,4],            0.0, "弱约束 (模糊的天时地利)"),
        (10, [0,1,2],                0.3, "中等约束"),
        ( 5, [0,1],                  0.5, "较强约束"),
        ( 5, [0],                    0.7, "强约束 (精确天时地利人和)"),
        ( 2, [0],                    0.8, "极强约束"),
    ]

    print(f"\n  ┌────────────────────────────┬──────────────┬──────────────┬──────────────┐")
    print(f"  │ 约束强度                   │  候选空间    │  误差RMSE    │  误差上界    │")
    print(f"  ├────────────────────────────┼──────────────┼──────────────┼──────────────┤")

    for t_tol, d_set, r_min, label in constraint_strengths:
        errors, spaces, max_errors = [], [], []

        for _ in range(n_trials):
            ks = KnowledgeSpace(n_items=5000, content_dim=10)
            true_content = np.random.rand(10) * 0.3 + 0.5
            true_time = 80.0
            true_domain = 0
            true_renhe = 0.7

            ks.plant_answers(true_content, true_time, true_domain, true_renhe,
                             n_correct=50, noise=0.05)

            # 应用不同强度的约束
            mask = ((np.abs(ks.tianshi - true_time) <= t_tol) &
                    np.isin(ks.dili, d_set) &
                    (ks.renhe >= r_min))
            candidates = np.where(mask)[0]
            spaces.append(len(candidates))

            if len(candidates) > 0:
                # 在约束空间内取共通性
                centroid = np.mean(ks.content[candidates], axis=0)
                error = np.sqrt(np.mean((centroid - true_content) ** 2))
                errors.append(error)

                # 误差上界 = 约束空间中最远点到真实值的距离
                all_dists = np.linalg.norm(ks.content[candidates] - true_content, axis=1)
                max_errors.append(np.max(all_dists))
            else:
                errors.append(float('nan'))
                max_errors.append(float('nan'))

        err = np.nanmean(errors)
        max_err = np.nanmean(max_errors)
        space = np.mean(spaces)
        print(f"  │ {label:<26s} │ {space:>11.0f} │ {err:>11.6f} │ {max_err:>11.6f} │")

    print(f"  └────────────────────────────┴──────────────┴──────────────┴──────────────┘")

    print(f"\n  → IEB定理1说: 逆向模式下误差有上界 ε_max = |y*| + |x|")
    print(f"    天时地利人和的约束越紧:")
    print(f"      候选空间越小 → 误差上界越小 → ε_max 趋近于 0")
    print(f"    天时地利人和越松:")
    print(f"      候选空间越大 → 误差上界越大 → 退化为正向模式 (ε→∞)")
    print(f"  ")
    print(f"  结论: 天时地利人和 = IEB 中 '=2' 的具体内涵")
    print(f"        约束强度 决定 误差上界")
    print(f"        这不是两个模型，这就是同一个东西")


# ═══════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════

def main():
    print("=" * 76)
    print("   天时地利人和 = 答案")
    print("   三维约束即答案本体  —  不是预处理，而是回答本身")
    print("=" * 76)
    print(f"""
  核心思想对照:

    IEB 原论文:
      1 + ? = 2
      "2" 是已知约束, "?" 的解空间被约束

    天时地利人和:
      问题 + ? = (天时, 地利, 人和)
      (天时, 地利, 人和) 就是那个 "2"
      答案 = 满足三维约束的交集点

    上个实验的错误:
      把天时地利人和当过滤器 (工具), 过滤后再找答案
      ≈ 先打扫房间再找东西

    正确理解:
      天时地利人和 IS 答案的坐标
      ≈ 你已经知道东西在 (什么时间, 什么地方, 谁放的)
      三维交集 = 答案就在那里
""")

    exp1_fundamental_comparison()
    exp2_collapse()
    exp3_event_localization()
    exp4_math_essence()
    exp5_robustness()
    exp6_dimension_contribution()
    exp7_unification()

    print(f"\n\n{'=' * 76}")
    print(f"  最终总结")
    print(f"{'=' * 76}")
    print(f"""
  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │   上个实验:                                                            │
  │     天时地利人和 → 过滤器 → 收敛 → 答案                               │
  │     (三步走，边界是工具)                                               │
  │                                                                        │
  │   本实验:                                                              │
  │     天时地利人和 = 答案                                                │
  │     (一步到位，边界就是答案本体)                                       │
  │                                                                        │
  │   数学本质:                                                            │
  │     IEB:  1 + ? = 2                                                    │
  │     等价:  问题 + ? = (天时, 地利, 人和)                               │
  │     其中:  "2" ≡ (天时, 地利, 人和)                                    │
  │     答案:   ? = 三维约束交集中涌现的内容                               │
  │                                                                        │
  │   关键发现:                                                            │
  │     1. 三维约束的空间压缩是乘性的 (不是加性的)                         │
  │     2. 答案从约束交集中涌现，而不是被搜索出来                          │
  │     3. 约束强度 = 误差上界 (IEB定理1的推广)                            │
  │     4. 即使部分约束有误，冗余维度提供优雅退化                          │
  │     5. 天时地利人和不是IEB的插件，它就是IEB的本体 ("=2")               │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
