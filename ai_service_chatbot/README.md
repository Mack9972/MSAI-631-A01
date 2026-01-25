# AI Service Chatbot (Microsoft LLM)

Rule-based chatbot with optional Microsoft Azure OpenAI fallback.

## Design notes
- Pure Python (no third-party dependencies required)
- Small, curated FAQ in `corpus.json` to avoid random answers
- Azure OpenAI used only when configured via environment variables

## Run
```bash
python3 chatbot.py
```

## Azure OpenAI configuration (optional)
Set these environment variables to enable the LLM fallback:

```bash
export AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
export AZURE_OPENAI_API_VERSION="2024-02-01"
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
