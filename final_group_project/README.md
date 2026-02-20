# Final Group Project (FastAPI + React)

Single-turn English input -> Spanish output, with optional English output. Backend now uses a document-based RAG knowledge base and can call Azure OpenAI for generation + translation when configured.

## Project Structure (MVC)
- `backend/app/controllers` - FastAPI routes
- `backend/app/services` - business logic (LLM, translation, rules)
- `backend/app/repositories` - data access (RAG index)
- `backend/app/models` - request/response schemas
- `frontend` - React UI
- `knowledge_base` - drop additional documents here (`.docx`, `.txt`, `.md`, `.json`, `.csv`)

## Backend Setup
```bash
cd "MSAI-631-A01/final_group_project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --app-dir backend
```

### .env (recommended)
Create or update `MSAI-631-A01/final_group_project/.env`:
```bash
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com"
AZURE_OPENAI_API_KEY="your-key"
AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
RAG_KB_PATHS="knowledge_base/LLM_Chatbot_Design_Report.docx,knowledge_base/Group_Project_Proposal_Full_RAG_Bilingual_Assistant.docx"
RAG_KB_DIR="knowledge_base"
CORS_ORIGINS="http://localhost:5173"
```

## RAG Knowledge Base
- The app indexes the two project `.docx` files by default.
- To expand the knowledge base, add files into `knowledge_base/` and ask questions. Supported formats:
  - `.docx`
  - `.txt`
  - `.md`
  - `.json` (with `question/answer` rows)
  - `.csv` (with `question,answer` columns)
- The backend auto-detects file changes and refreshes the index on the next request.

## Frontend Setup
```bash
cd "MSAI-631-A01/final_group_project/frontend"
npm install
npm run dev
```

Optional override for API base URL:
```bash
export VITE_API_URL="http://localhost:8000"
```

## API
- `POST /api/translate`

Request:
```json
{ "text": "What is a chatbot?", "include_english": true }
```

Response:
```json
{ "spanish": "...", "english": "..." }
```
