# tng-data-persona

Dedicated persona repo for Data.

This repo stays focused on Data-specific prompting, examples, and evaluation. The repo keeps its own source copy at `data/raw/dialogue.jsonl`, and the extractor turns that corpus into Data-specific training examples.

## Extractor

```bash
python -m scripts.extract_data_persona
```

Outputs:

- `data/tng_data_train.jsonl`
- `data/tng_data_eval.jsonl`
- `data/tng_data_counterexamples.jsonl`
- `data/tng_data_summary.json`
- `data/data_extract_report.md`

## Scope

- Data voice prompt
- Data-specific examples and counterexamples
- Evaluation notes for literal precision and reasoning quality
- Links back to the shared TNG corpus
