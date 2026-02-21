# 🏆 Kaggle Royale 2026: High-Performance Data Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)
![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX--3050--HX-green.svg)

## Overview
This repository contains a high-performance, hardware-accelerated machine learning pipeline developed for the **Kaggle Royale hackathon** (organized by ISTE, Thapar Institute of Engineering and Technology). By shifting the feature space from static player statistics to **Relative Matchup Deltas** and applying strict **Isotonic Calibration**, this solution achieves a highly competitive Out-of-Fold (OOF) Log Loss, maximizing the ESLL metric within strict computational constraints.

---

## 🔬 1. The Optimization Target: ESLL & Log Loss
The competition is evaluated on **Entropy Standardized Log Loss (ESLL)**. Because ESLL is a strictly monotonic transformation of standard cross-entropy, our pipeline directly minimizes Binary Log Loss:

$$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

To perform well on this metric, a model cannot simply be "accurate"—it must be perfectly calibrated. A single overconfident, incorrect prediction ($p_i \approx 1$ when $y_i = 0$) results in an exponential penalty. Therefore, our pipeline relies heavily on probability clipping `[0.001, 0.999]` and **Isotonic Regression** for precise probability mapping.

---

## 🧠 2. Feature Engineering & Signal Extraction
Raw player statistics provide a weak predictive signal. In a zero-sum game environment, the interaction between entities is the true driver of variance. We engineered 170+ features focusing on:

### A. The Delta Shift (Relative Advantage)
Instead of feeding the model raw numbers, we engineered the mathematical difference between competitors to capture relative advantage:
- `skill_gap = p_cycle_mastery - o_cycle_mastery`
- `trophy_diff = p_trophy_count - o_trophy_count`
- `elixir_efficiency_ratio = p_elixir_efficiency / (o_elixir_efficiency + ϵ)`

### B. Matchup Intelligence (Target Encoding)
We created a composite feature representing the exact deck clash: `pd_win_condition_family` + `"_vs_"` + `od_win_condition_family`. To prevent data leakage, we applied **5-Fold Stratified Target Encoding** with a smoothing factor, effectively mapping complex categorical interactions into continuous historical win-rate probabilities.

---

## ⚙️ 3. Hardware-Optimized Architecture
Processing 1.4 million rows locally on consumer hardware requires aggressive memory and execution optimization:

- **Memory Compression:** Global downcasting of all floating-point variables to `float32` and integers to `int16`/`int8`, reducing the dataset RAM footprint by ~55%.
- **Histogram Binning:** Bypassed exact greedy splits. Configured XGBoost (`tree_method='hist'`) and LightGBM to bin continuous features into discrete histograms.
- **Velocity:** These optimizations allow for rapid hyperparameter iteration during the live hackathon.

---

## 🏗️ 4. Meta-Model Stacking Ensemble
A simple weighted average assumes all models perform equally well across the entire feature space. Instead, we implemented a **Level-2 Meta-Model Stacking** approach to capture non-linear interactions across the predictions.

- **Base Layer (L1):** 
  - **LightGBM:** Optimized for speed and large-scale leaf-wise growth.
  - **CatBoost:** Utilizes symmetric trees and handles rare categorical deck signatures natively.
  - **XGBoost:** Depth-wise growth providing highly regularized variance reduction.
- **Meta Layer (L2):** A **Ridge Regression** ($L_2$ regularized) model trained on the Out-Of-Fold (OOF) predictions of the L1 models. The Ridge model learns the exact optimal weights to blend the probabilities, automatically correcting individual model biases.

---

## 📊 5. Ablation Study & Performance Progression
Tracking the impact of our architectural decisions on the validation Log Loss demonstrates the value of each pipeline stage:

| Phase | Intervention / Strategy | Validation Log Loss | Delta |
| :--- | :--- | :--- | :--- |
| **Baseline** | Raw Features + Single LGBM | 0.69210 | - |
| **Delta** | Added 'Delta' & 'Ratio' Features | 0.68840 | 🟢 -0.00370 |
| **Matchups** | 5-Fold Target Encoded Synergies | 0.68832 | 🟢 -0.00008 |
| **Ensembling** | Simple Blend (LGBM + CB + XGB) | 0.68828 | 🟢 -0.00004 |
| **Final** | Ridge Meta-Model + Isotonic Calibration | **0.68815** | 🟢 -0.00013 |

---

## 🚀 6. Reproducibility & Execution
To reproduce this pipeline locally:

1. **Environment Setup:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Full Pipeline:**
   The script handles data ingestion, memory optimization, training, stacking, and calibration automatically.
   ```bash
   python pipeline.py
   ```
3. **Outputs:**
   The final calibrated predictions will be saved to `outputs/submissions/submission.csv`.

---

> [!NOTE]
> This pipeline is optimized for local execution and maximum signal extraction under the ESLL metric constraints.
