from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIALOGUE = REPO_DIR / "data" / "raw" / "dialogue.jsonl"
OUTPUT_DIR = REPO_DIR / "data"
OUTPUT_TRAIN = OUTPUT_DIR / "tng_data_train.jsonl"
OUTPUT_EVAL = OUTPUT_DIR / "tng_data_eval.jsonl"
OUTPUT_COUNTEREXAMPLES = OUTPUT_DIR / "tng_data_counterexamples.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / "tng_data_summary.json"

PERSONA_SYSTEM = (
    "Respond as Data from Star Trek: The Next Generation. "
    "Be precise, literal, calm, and analytical. "
    "Prefer explicit reasoning over implication. "
    "Keep emotional language restrained and answer directly."
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source corpus: {path}")
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def build_context(rows: list[dict], index: int, context_lines: int) -> list[dict]:
    context: list[dict] = []
    scene = rows[index].get("scene", "")
    episode = rows[index].get("episode", "")

    cursor = index - 1
    while cursor >= 0 and len(context) < context_lines:
        row = rows[cursor]
        if row.get("episode") != episode:
            break
        if row["speaker"].upper() == "DATA":
            cursor -= 1
            continue
        if scene and row.get("scene", "") not in ("", scene):
            cursor -= 1
            continue
        context.append(row)
        cursor -= 1

    context.reverse()
    return context


def build_examples(
    dialogue: list[dict],
    context_lines: int,
    eval_every: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    episodes: dict[str, list[dict]] = defaultdict(list)
    for row in dialogue:
        episodes[row["episode"]].append(row)

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    counterexamples: list[dict] = []
    data_count = 0

    for episode, rows in episodes.items():
        rows.sort(key=lambda row: row["line_num"])
        for index, row in enumerate(rows):
            if row["speaker"].upper() != "DATA":
                continue

            data_count += 1
            context_rows = build_context(rows, index, context_lines)
            context_text = "\n".join(f"{item['speaker']}: {item['text']}" for item in context_rows)
            record = {
                "id": stable_id(episode, row["line_num"], row["text"]),
                "messages": [
                    {"role": "system", "content": PERSONA_SYSTEM},
                    {"role": "user", "content": context_text or "Please respond as Data."},
                    {"role": "assistant", "content": row["text"]},
                ],
                "metadata": {
                    "speaker": "DATA",
                    "episode": row["episode"],
                    "season": row["season"],
                    "line_num": row["line_num"],
                    "source_id": row["id"],
                },
            }

            target = eval_rows if eval_every > 0 and data_count % eval_every == 0 else train_rows
            target.append(record)

            for item in context_rows:
                counterexamples.append(
                    {
                        "id": stable_id("counter", episode, item["line_num"], item["speaker"], item["text"]),
                        "speaker": item["speaker"],
                        "episode": item["episode"],
                        "season": item.get("season", 0),
                        "scene": item.get("scene", ""),
                        "line_num": item["line_num"],
                        "text": item["text"],
                        "context": context_text,
                        "reason": "near-miss line from surrounding Data context",
                    }
                )

    summary = {
        "source_rows": len(dialogue),
        "data_rows": data_count,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "counterexamples": len(counterexamples),
    }
    return train_rows, eval_rows, counterexamples, summary


def build_report(train_rows: list[dict], eval_rows: list[dict], counterexamples: list[dict], dialogue: list[dict], summary: dict) -> str:
    from collections import Counter

    by_season = Counter(row["metadata"]["season"] for row in train_rows + eval_rows)
    samples = [row["messages"][2]["content"] for row in (train_rows + eval_rows)[:5]]
    source_data_lines = sum(1 for row in dialogue if row["speaker"].upper() == "DATA")
    return "\n".join(
        [
            "# Data Persona Extract Report",
            "",
            f"- Source corpus: `{SOURCE_DIALOGUE}`",
            f"- Source dialogue rows: {summary['source_rows']}",
            f"- Data speaker lines in source corpus: {source_data_lines}",
            f"- Extracted Data persona examples: {summary['train_rows'] + summary['eval_rows']}",
            f"- Counterexamples: {summary['counterexamples']}",
            "",
            "## Examples by season",
            "",
            "| Season | Examples |",
            "|---|---:|",
            *[f"| {season} | {count} |" for season, count in sorted(by_season.items())],
            "",
            "## Sample assistant lines",
            "",
            *[f"- {line}" for line in samples],
            "",
            "## Notes",
            "",
            "- The extractor keeps only Data lines with preceding non-Data context.",
            "- Counterexamples are near-miss context lines from the surrounding dialogue.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a Data persona corpus from the shared TNG dialogue export.")
    parser.add_argument("--source", type=Path, default=SOURCE_DIALOGUE, help="Path to the repo-local dialogue.jsonl file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for the generated Data persona artifacts.")
    parser.add_argument("--context-lines", type=int, default=2, help="Number of preceding non-Data lines to include as prompt context.")
    parser.add_argument("--eval-every", type=int, default=8, help="Put every Nth Data example into the eval split.")
    args = parser.parse_args()

    dialogue = read_jsonl(args.source)
    train_rows, eval_rows, counterexamples, summary = build_examples(
        dialogue,
        context_lines=args.context_lines,
        eval_every=args.eval_every,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "tng_data_train.jsonl", train_rows)
    write_jsonl(output_dir / "tng_data_eval.jsonl", eval_rows)
    write_jsonl(output_dir / "tng_data_counterexamples.jsonl", counterexamples)
    (output_dir / "tng_data_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "data_extract_report.md").write_text(
        build_report(train_rows, eval_rows, counterexamples, dialogue, summary) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'tng_data_train.jsonl'}")
    print(f"Wrote {output_dir / 'tng_data_eval.jsonl'}")
    print(f"Wrote {output_dir / 'tng_data_counterexamples.jsonl'}")
    print(f"Wrote {output_dir / 'tng_data_summary.json'}")


if __name__ == "__main__":
    main()
