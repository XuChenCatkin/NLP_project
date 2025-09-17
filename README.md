# NLP_project (refactored)

A clean, reproducible implementation of our NLP pipelines (embedding, retrieval, NER, QA, and question generation).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]

pytest -q
nlp-project --help
```

## Structure
Library in `src/nlp_project/`, scripts in `scripts/`.

## Data & Artifacts
Large files (indexes, checkpoints) are **not** tracked by Git. Use Git LFS or provide a download script. See `data/README.md`.

## Citation
See `CITATION.cff`.
