"""Build an executed notebook from the reproducible homework script."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


here = Path(__file__).resolve().parent
source = (here / "solve_homework.py").read_text(encoding="utf-8")
source = source.replace('HERE = Path(__file__).resolve().parent', 'HERE = Path.cwd()')
boundaries = [
    'print("\\nQ1: Single-feature AUC (direction corrected)")',
    'def train_and_predict(train, validation, c=1.0):',
    'thresholds = np.arange(101) / 100',
    'def cross_validation(c):',
    'print("\\nQ6: Cross-validation by C")',
]
parts = []
for boundary in boundaries:
    before, source = source.split(boundary, 1)
    parts.append(before)
    source = boundary + source
parts.append(source)

headings = [
    "## Data preparation",
    "## Question 1: ROC AUC feature importance",
    "## Question 2: Logistic regression validation AUC",
    "## Questions 3 and 4: Precision, recall, and F1",
    "## Question 5: Five-fold cross-validation",
    "## Question 6: Hyperparameter tuning",
]
cells = [
    nbf.v4.new_markdown_cell(
        "# Homework 4: Evaluation Metrics for Classification\n\n"
        "2026 lead-scoring data; all results use the assignment's exact settings."
    )
]
for index, (heading, part) in enumerate(zip(headings, parts)):
    cells.extend([nbf.v4.new_markdown_cell(heading), nbf.v4.new_code_cell(part.strip())])
    if index == 3:
        cells.append(nbf.v4.new_markdown_cell("![Validation precision and recall](precision_recall.png)"))

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook.metadata.language_info = {"name": "python"}
NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(here)}}).execute()
nbf.write(notebook, here / "homework4.ipynb")
