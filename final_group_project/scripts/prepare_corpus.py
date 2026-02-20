#!/usr/bin/env python3
"""
Build corpus.json from DailyDialog + Persona-Chat using HuggingFace datasets.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def add_pair(pairs: list[dict], question: str, answer: str) -> None:
    question = normalize_text(question)
    answer = normalize_text(answer)
    if question and answer and question != answer:
        pairs.append({"question": question, "answer": answer})


def build_from_daily_dialog(dataset_id: str, split: str, trust_remote_code: bool) -> list[dict]:
    pairs: list[dict] = []
    dataset = load_dataset(dataset_id, split=split, trust_remote_code=trust_remote_code)
    current_id = None
    previous = None
    for row in dataset:
        dialog_id = row.get("dialog_id")
        utterance = row.get("utterance")
        if dialog_id != current_id:
            current_id = dialog_id
            previous = None
        if previous is not None and utterance:
            add_pair(pairs, previous, utterance)
        if utterance:
            previous = utterance
    return pairs


def build_from_persona_chat(dataset_id: str, split: str, trust_remote_code: bool) -> list[dict]:
    pairs: list[dict] = []
    dataset = load_dataset(dataset_id, split=split, trust_remote_code=trust_remote_code)
    for row in dataset:
        question = row.get("question")
        answer = row.get("answer")
        if question and answer:
            add_pair(pairs, question, answer)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chatbot corpus from HF datasets.")
    parser.add_argument("--out", default="corpus.json", help="Output JSON file path.")
    parser.add_argument("--max-pairs", type=int, default=50000, help="Max total pairs to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--daily-dataset",
        default="pixelsandpointers/better_daily_dialog",
        help="DailyDialog dataset id.",
    )
    parser.add_argument("--daily-split", default="train", help="DailyDialog split.")
    parser.add_argument("--persona-dataset", default="yatsby/persona_chat", help="Persona-Chat dataset id.")
    parser.add_argument("--persona-split", default="train", help="Persona-Chat split.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow datasets with custom code.")
    args = parser.parse_args()

    pairs = []
    pairs.extend(build_from_daily_dialog(args.daily_dataset, args.daily_split, args.trust_remote_code))
    pairs.extend(build_from_persona_chat(args.persona_dataset, args.persona_split, args.trust_remote_code))

    random.Random(args.seed).shuffle(pairs)
    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = pairs[: args.max_pairs]

    out_path = Path(args.out)
    out_path.write_text(json.dumps(pairs, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()
