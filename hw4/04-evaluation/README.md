# Homework 4: Evaluation Metrics for Classification

This solution uses the pinned 2026 lead-scoring dataset and the exact split,
missing-value treatment, model parameters, and five-fold cross-validation from
the [assignment](homework.md). Run `python solve_homework.py` from any directory
with pandas, NumPy, scikit-learn, and Matplotlib installed.

| Question | Answer | Computed result |
| --- | --- | --- |
| 1 | `lead_score` | ROC AUC 0.788225 (direction corrected) |
| 2 | `0.732` | Validation ROC AUC 0.731822 |
| 3 | `0.63` | Precision 0.732759, recall 0.725256 |
| 4 | `0.41` | Maximum F1 0.757616 |
| 5 | `0.007` | Five-fold AUC standard deviation 0.006853 |
| 6 | `0.001` | Best mean AUC 0.749 (std 0.010) |

The [notebook](homework4.ipynb) shows the calculations and the
[precision/recall plot](precision_recall.png). The separate script runs the
same calculation without Jupyter.
