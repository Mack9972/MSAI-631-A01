# Traditional Chatbot

Simple, rule-based chatbot (no LLMs).

## Run
```bash
python3 chatbot.py
```

## Build a larger corpus (Persona-Chat + DailyDialog)
Requires: `datasets` library and network access to download the data once.
Defaults to `pixelsandpointers/better_daily_dialog` and `yatsby/persona_chat`.

```bash
python3 scripts/prepare_corpus.py --out corpus.json --max-pairs 50000
```

## Example prompts
- hello
- my name is Alex
- define intent
- what is a chatbot
- what time is it
- help
- exit
