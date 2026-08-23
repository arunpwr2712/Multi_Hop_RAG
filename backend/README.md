# FastAPI Backend

This backend wraps the existing `multi_hop_causal_rag` logic without changing model behavior.

## Run

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /api/health`
- `POST /api/chat`
- `GET /api/graph`
- `GET /api/last-result`
- `POST /api/reset`
