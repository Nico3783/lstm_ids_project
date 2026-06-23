# Chapter 1 Restructuring Plan

## Current State
Chapter 1 has 9 subsections (1.1–1.9) with ~120 lines of text. No SVG images used. Math formulas not numbered.

## Target Structure (5 sections)

### 1.1 Preamble (~15 lines)
- Opening paragraph setting the stage for cybersecurity in Nigeria
- Brief overview of the IDS challenge
- Introduce the research focus: LSTM-based deep learning for intrusion detection
- **Content from**: Current 1.1 (first 2 paragraphs), 1.6 (significance — split with Motivation)
- **Final paragraph**: Brief mention of chapter structure (from 1.9)

### 1.2 Background (~25 lines)
- Network security landscape in Nigeria (banking, telecom, government)
- Evolution of IDS: signature-based → anomaly-based → ML → deep learning
- The temporal dependency problem in conventional classifiers
- **Content from**: Current 1.1 (paragraphs 3–5), 1.2 (problem statement), 1.3 (research questions)
- **No images** — conceptual overview only

### 1.3 Research Motivation (~20 lines)
- Why LSTM specifically (temporal modeling capability)
- Limitations of existing approaches in Nigerian context
- Research gap: no comprehensive LSTM-based IDS study across 3 benchmark datasets
- **Content from**: Current 1.2 (problem statement), 1.4.1 (aim), 1.5 (hypotheses), 1.6 (significance — split with Preamble)
- **No images** — conceptual overview only

### 1.4 Objectives (~15 lines)
- Research Aim (from 1.4.1)
- Research Objectives (from 1.4.2) — numbered list
- Research Questions (from 1.3) — integrated as sub-questions under objectives
- **Hypotheses** (from 1.5) — added after objectives

### 1.5 Methodology (~25 lines) — **NEW SECTION**
- Brief overview of the experimental approach
- Datasets: NSL-KDD, CICIDS2017, UNSW-NB15
- Preprocessing pipeline (introduce the 5-stage process)
- LSTM model architecture (introduce the stacked architecture)
- Evaluation metrics (accuracy, precision, recall, F1-score)
- **No images** — conceptual introduction only
- **Numbered equations**: Introduce key formulas with equation numbers (1)–(N)
- Reference to Chapter 3 for full mathematical detail and diagrams

### Appendix A: Scope and Limitations (from 1.7)
### Appendix B: Definitions of Key Terms (from 1.8) and diagrams

### Remaining Content (moved elsewhere)
- 1.6 Significance → merged into Preamble and Background
- 1.7 Scope and Limitations → moved to Chapter 3 (Methodology) or Appendix
- 1.8 Definitions of Key Terms → moved to Appendix
- 1.9 Structure of the Study → moved to Introduction of Chapter 2 or kept as final paragraph

---

## Mathematical Formula Numbering Strategy

All formulas in Chapter 1 will use equation numbers. These same equations appear in Chapter 3 but are re-introduced here with brief explanations.

### Equations to include in Chapter 1:

| Eq# | Formula | Location |
|-----|---------|----------|
| (1) | Min-Max Scaling | §1.5 Methodology |
| (2) | Sliding Window Construction | §1.5 Methodology |
| (3) | One-Hot Encoding | §1.5 Methodology |
| (4) | Forget Gate | §1.5 Methodology |
| (5) | Input Gate | §1.5 Methodology |
| (6) | Candidate Cell State | §1.5 Methodology |
| (7) | Cell State Update | §1.5 Methodology |
| (8) | Output Gate | §1.5 Methodology |
| (9) | Hidden State | §1.5 Methodology |
| (10) | Softmax Activation | §1.5 Methodology |
| (11) | Categorical Cross-Entropy Loss | §1.5 Methodology |
| (12) | Accuracy | §1.5 Methodology |
| (13) | Precision | §1.5 Methodology |
| (14) | Recall | §1.5 Methodology |
| (15) | F1-Score | §1.5 Methodology |

---

## SVG Image Usage Plan

**Chapter 1** — NO SVG images. Methodology section is a conceptual introduction only.

**Chapter 3** — All 5 SVGs used here:

| Image | Location in Chapter 3 | Caption |
|-------|----------------------|---------|
| `methodology_flowchart.svg` | §3.2 Research Design | Figure 3.1: Research methodology flowchart |
| `preprocessing_pipeline.svg` | §3.5.2 Preprocessing | Figure 3.2: Data preprocessing pipeline |
| `lstm_model_architecture.svg` | §3.5.3 LSTM Architecture | Figure 3.3: Proposed LSTM model architecture |
| `system_architecture.svg` | §3.1 Introduction | Figure 3.4: System architecture overview |
| `evaluation_metrics_framework.svg` | §3.3.7 Evaluation Metrics | Figure 3.5: Evaluation metrics framework |

---

## Content Merge Mapping

| Current Section | Destination | Content to Keep |
|----------------|-------------|-----------------|
| 1.1 Background | 1.1 Preamble + 1.2 Background | Split: paragraphs 1–2 → Preamble; paragraphs 3–5 → Background |
| 1.2 Problem Statement | 1.2 Background + 1.3 Motivation | IDS limitations → Background; ML limitations → Motivation |
| 1.3 Research Questions | 1.4 Objectives | Integrated as sub-questions |
| 1.4 Aim & Objectives | 1.4 Objectives | Kept as-is |
| 1.5 Hypotheses | 1.4 Objectives | Added after objectives |
| 1.6 Significance | 1.1 Preamble + 1.3 Motivation | Split between Preamble and Motivation |
| 1.7 Scope & Limitations | Appendix A | Moved to Appendix A |
| 1.8 Definitions | Appendix B | Moved to Appendix B |
| 1.9 Structure | 1.1 Preamble (final paragraph) | Brief mention of chapter structure |

---

## Equation Numbering Format

Use LaTeX-style equation numbering that pandoc can convert to docx:

```markdown
$$x'_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}} \tag{1}$$
```

This ensures equations are numbered (1)–(15) in the docx output.

---

## Estimated Length

Current Chapter 1: ~120 lines (~4,800 words)
New Chapter 1: ~110 lines (~4,400 words) — more concise, better structured
Appendices: ~30 lines (~1,200 words) — Scope/Definitions moved here

---

## Verification Checklist

- [ ] No SVG images in Chapter 1 (all 5 SVGs reserved for Chapter 3)
- [ ] All 15 equations numbered with \tag{}
- [ ] All content from original 9 sections preserved (merged into new structure)
- [ ] Cross-references to Chapter 3 for full methodology detail and diagrams
- [ ] Humanized academic writing maintained
- [ ] Consistent citation format throughout
