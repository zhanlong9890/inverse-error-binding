# Inverse Error Binding (IEB) — AI Hallucination Suppression Framework

## 答案约束优于问题求解：逆向误差绑定框架

[![Paper](https://img.shields.io/badge/Paper-Markdown-blue)](paper.md)
[![arXiv](https://img.shields.io/badge/arXiv-2026-red)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

### Core Idea / 核心思想

> **$1 + ? = 2$ is safer than $1 + 1 = ?$**

- **Forward mode** ($1+1=?$): Output space is unbounded → Error can be infinite
- **Inverse mode** ($1+?=2$): Output is constrained by the answer → Error is bounded and verifiable

Applied to AI: instead of making models understand questions better, **use answer structure to constrain output error** — eliminating hallucination through multi-source convergence.

---

### Key Results / 核心结果

| Scale | Big-Data Filtering Precision | Answer Convergence Precision |
|-------|-----|-----|
| 1K | 0.000 | **1.000** |
| 10K | 0.000 | **1.000** |
| 100K | 0.000 | **1.000** |
| 1M | 0.000 | **1.000** |

100% precision at all scales. The advantage is **certainty**, not speed.

---

### Repository Structure / 仓库结构

```
IEB-paper/
├── README.md           ← You are here
├── paper.md            ← Full paper (Markdown, bilingual)
├── experiment_code.py  ← Complete experimental validation (4 experiments)
└── latex/
    └── main.tex        ← LaTeX version for arXiv submission
```

---

### Run Experiments / 运行实验

```bash
pip install numpy
python experiment_code.py
```

Outputs 4 experiments:
1. **Precision vs Scale** — 100% precision from 1K to 1M
2. **Noise Elimination** — Matches theoretical $1/\sqrt{n}$ prediction
3. **Forward vs Inverse Error** — Forward mode error 6497× larger
4. **Convergence Speed** — 20 sources sufficient even at high noise (σ=0.5)

---

### Relation to Existing Work / 与现有工作的关系

This framework provides the **epistemological foundation** for why methods like Self-Consistency (Wang et al., ICLR 2023) work. Self-Consistency is a special case of Answer Convergence at the engineering level. IEB explains *why*: **it's not about voting — it's about error binding.**

---

### Citation / 引用

```bibtex
@article{maxur2026ieb,
  title={Answer-Constrained Reasoning Outperforms Question-Based Solving: 
         An Inverse Error-Binding Framework for AI Hallucination Suppression},
  author={MAXUR},
  year={2026},
  note={Independent research}
}
```

---

### Philosophy / 哲学延伸

> Science is not about finding the answer.  
> Science is about figuring out the path to the answer, once you know where it is.
>
> 科学不是关于找到答案。科学是关于知道答案在哪里之后，搞清楚通往答案的路。

---

**Author: MAXUR** | February 2026 | CC BY 4.0
