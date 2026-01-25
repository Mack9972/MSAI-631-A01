# Traditional Chatbot

Simple, rule-based chatbot (no LLMs).

## Design notes
- Pure Python (no third-party dependencies required)
- Small, curated FAQ in `corpus.json` to avoid random answers

## Run
```bash
python3 chatbot.py
```

## Optional: build a larger corpus from HuggingFace datasets
Requires: `datasets` and network access to download the data once.

```bash
python3 scripts/prepare_corpus.py --out corpus.json --max-pairs 5000
```

## Example prompts
- hello
- my name is Alex
- define intent
- what is a chatbot
- what time is it
- help
- exit
