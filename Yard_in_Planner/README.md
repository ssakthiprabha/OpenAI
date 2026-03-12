# Container Yard Planner — YARD CHECK
## Python/Flask + GenAI Backend

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run the server
python app.py

# 4. Open in browser
open http://localhost:5000
```

### Project Structure

```
yard_planner/
├── app.py              ← Flask backend (REST API + AI endpoint)
├── requirements.txt
├── README.md
└── templates/
    └── index.html      ← Full UI (served by Flask)
```

---

### REST API Endpoints

| Method | Path                          | Description                          |
|--------|-------------------------------|--------------------------------------|
| GET    | `/api/zones`                  | All yard zones + slot lists          |
| GET    | `/api/trailers`               | All trailers                         |
| POST   | `/api/trailers`               | Add a new trailer                    |
| PUT    | `/api/trailers/<id>`          | Update trailer fields                |
| DELETE | `/api/trailers/<id>`          | Remove trailer                       |
| POST   | `/api/move`                   | Move trailer to slot (or pool)       |
| POST   | `/api/ai`                     | Ask YARD-AI a question               |

#### POST `/api/trailers` body
```json
{
  "id": "SPAN 401748",
  "dest": "FBX",
  "status": "loaded",
  "date": "9/5",
  "notes": "(L)",
  "double": false,
  "slot": null
}
```

#### POST `/api/move` body
```json
{ "id": "SPAN 401748", "slot": "Y-130" }
```
Pass `"slot": null` to move trailer back to the unassigned pool.

#### POST `/api/ai` body
```json
{
  "message": "Which trailers are in detention?",
  "history": []
}
```
Returns `{ "reply": "..." }`. Pass prior turns in `history` for multi-turn conversation.

---

### YARD-AI Features

The AI assistant has a **live snapshot** of the yard injected into every request:
- All zone occupancy counts
- Every trailer's ID, status, destination, slot
- Unassigned pool

Example questions:
- *"Give me a shift handover summary"*
- *"Where is SPAN 401748?"*
- *"Which FBX trailers are ready?"*
- *"What slots are free in the West Side?"*
- *"Any detention trailers I should call about?"*

---

### Production Notes

- Replace the in-memory `trailers` list with **SQLite** (via `flask-sqlalchemy`) or **PostgreSQL** for persistence.
- Add authentication (Flask-Login or JWT) before exposing publicly.
- Set `debug=False` and use `gunicorn` or `waitress` for production serving.
