"""
Container Yard Planner — Flask Backend
Serves the planner UI and provides REST API + AI assistant endpoint.
"""

import os
import json
import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
import anthropic

app = Flask(__name__)

# ─────────────────────────────────────────────
# In-memory store (swap for SQLite/Postgres in prod)
# ─────────────────────────────────────────────
ZONES = [
    {"id": "y-main",   "label": "EAST SIDE MIDDLE — Y DOORS (127–137)",
     "slots": ["Y-127","Y-128","Y-129","Y-130","Y-131","Y-132","Y-133","Y-134","Y-135","Y-136","Y-137"]},
    {"id": "y-ext",    "label": "EAST SIDE — Y DOORS (145–155)",
     "slots": ["Y-145","Y-146","Y-147","Y-148","Y-149","Y-150","Y-151","Y-152","Y-153","Y-154","Y-155"]},
    {"id": "x-doors",  "label": "WEST SIDE — X DOORS (145–155)",
     "slots": ["X-145","X-146","X-147","X-148","X-149","X-150","X-151","X-152","X-153","X-154","X-155"]},
    {"id": "dr-doors", "label": "DRIVE-IN DOORS (DR-34 to DR-39)",
     "slots": ["DR-34","DR-35","DR-36","DR-37","DR-38","DR-39"]},
    {"id": "y-lower",  "label": "YARD EXTENSIONS (Y-71 to Y-73)",
     "slots": ["Y-71","Y-72","Y-73"]},
]

trailers = [
    {"id":"SPAN 400112",   "dest":"VALLEY","status":"loaded",   "date":"9/5",  "notes":"(L)",                    "double":False,"slot":"Y-127"},
    {"id":"SPAN 401749",   "dest":"VALLEY","status":"loaded",   "date":"9/3",  "notes":"(L)",                    "double":False,"slot":"Y-128"},
    {"id":"SPAN 410204",   "dest":"VALLEY","status":"loaded",   "date":"9/10", "notes":"(L)",                    "double":False,"slot":"Y-129"},
    {"id":"SPAN 401748",   "dest":"FBX",   "status":"loaded",   "date":"8/19", "notes":"E (L)",                  "double":False,"slot":"Y-130"},
    {"id":"MATU 278383",   "dest":"FBX",   "status":"transit",  "date":"8/18", "notes":"E (T)",                  "double":False,"slot":"Y-131"},
    {"id":"SPAN 401701",   "dest":"FBX",   "status":"loaded",   "date":"9/10", "notes":"(L)",                    "double":False,"slot":"Y-132"},
    {"id":"MATU 265682",   "dest":"FBX",   "status":"transit",  "date":"8/18", "notes":"E (T)",                  "double":False,"slot":"Y-133"},
    {"id":"MATU 272905",   "dest":"FBX",   "status":"empty",    "date":"8/18", "notes":"E (L)",                  "double":False,"slot":"Y-134"},
    {"id":"PAFX 500915",   "dest":"FBX",   "status":"loaded",   "date":"9/10", "notes":"(L)",                    "double":False,"slot":"Y-135"},
    {"id":"PAFX 500910",   "dest":"FBX",   "status":"loaded",   "date":"9/3",  "notes":"(L)",                    "double":False,"slot":"Y-136"},
    {"id":"MATU 272905-B", "dest":"FBX",   "status":"empty",    "date":"8/18", "notes":"E (L)",                  "double":False,"slot":"Y-137"},
    {"id":"TCNU 695693",   "dest":"FBX",   "status":"transit",  "date":"",     "notes":"E (T)",                  "double":False,"slot":"Y-154"},
    {"id":"MATU 520267",   "dest":"VALLEY","status":"ready",    "date":"",     "notes":"E (L)(R)",               "double":False,"slot":"Y-155"},
    {"id":"SPAN 401801",   "dest":"FBX",   "status":"transit",  "date":"",     "notes":"E (T)",                  "double":False,"slot":"X-154"},
    {"id":"MATU 274401",   "dest":"FBX",   "status":"empty",    "date":"",     "notes":"E (L)",                  "double":False,"slot":"X-155"},
    {"id":"MSTS 228123",   "dest":"KODIAK","status":"detention","date":"",     "notes":"FOR KODIAK & TRUCK 132 ★","double":False,"slot":None},
    {"id":"BILLS HOLD",    "dest":"",      "status":"hold",     "date":"",     "notes":"Dave Cate",              "double":False,"slot":None},
    {"id":"BILLS HOLD 325","dest":"",      "status":"hold",     "date":"",     "notes":"Dave Cate (325)",        "double":False,"slot":None},
    {"id":"TCNU 562159",   "dest":"FBX",   "status":"backhaul", "date":"",     "notes":"FBX L/H",                "double":False,"slot":None},
    {"id":"SPAN 401811",   "dest":"FBX",   "status":"backhaul", "date":"",     "notes":"FBX L/H",                "double":False,"slot":None},
    {"id":"PAFX 500947",   "dest":"VALLEY","status":"backhaul", "date":"",     "notes":"E (L) 4 sale ty",        "double":False,"slot":None},
    {"id":"MSTS 228124",   "dest":"KODIAK","status":"transit",  "date":"",     "notes":"FOR KODIAK",             "double":False,"slot":None},
    {"id":"W 123",         "dest":"",      "status":"transit",  "date":"",     "notes":"W/ EXP DOT",             "double":False,"slot":None},
    {"id":"MATU 513648",   "dest":"",      "status":"transit",  "date":"",     "notes":"(T) own",                "double":False,"slot":None},
    {"id":"MATU 513603",   "dest":"",      "status":"transit",  "date":"",     "notes":"(T) own",                "double":False,"slot":None},
    {"id":"MATU 273332",   "dest":"VALLEY","status":"loaded",   "date":"",     "notes":"VALLEY",                 "double":False,"slot":None},
    {"id":"PAFX 400028",   "dest":"VALLEY","status":"loaded",   "date":"",     "notes":"VALLEY L/H (L)",         "double":False,"slot":None},
    {"id":"DBCU 945338",   "dest":"S.O.C.","status":"loaded",   "date":"8/18", "notes":"VALLEY (L)",             "double":False,"slot":None},
    {"id":"PAFX 400022",   "dest":"",      "status":"empty",    "date":"",     "notes":"AWNING",                 "double":False,"slot":None},
    {"id":"400041 STRIP",  "dest":"",      "status":"empty",    "date":"",     "notes":"STRIP",                  "double":False,"slot":None},
]

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def _find(tid):
    return next((t for t in trailers if t["id"] == tid), None)

def _snapshot():
    """Return a compact yard summary string for the AI."""
    from collections import Counter
    stats = Counter(t["status"] for t in trailers)
    zones_occ = []
    for z in ZONES:
        occ = [t for t in trailers if t["slot"] in z["slots"]]
        zones_occ.append(f'  {z["label"]}: {len(occ)}/{len(z["slots"])} occupied')
    unassigned = [t for t in trailers if not t["slot"]]
    return (
        f"Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Total trailers: {len(trailers)}\n"
        f"Status counts: {dict(stats)}\n"
        f"Zone occupancy:\n" + "\n".join(zones_occ) + "\n"
        f"Unassigned pool ({len(unassigned)}): "
        + ", ".join(f'{t["id"]}({t["status"]})' for t in unassigned)
    )

# ─────────────────────────────────────────────
# Routes — UI
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ─────────────────────────────────────────────
# Routes — REST API
# ─────────────────────────────────────────────
@app.route("/api/zones")
def get_zones():
    return jsonify(ZONES)

@app.route("/api/trailers", methods=["GET"])
def get_trailers():
    return jsonify(trailers)

@app.route("/api/trailers", methods=["POST"])
def add_trailer():
    data = request.json
    tid = data.get("id", "").strip().upper()
    if not tid:
        return jsonify({"error": "Trailer ID required"}), 400
    if _find(tid):
        return jsonify({"error": "Trailer ID already exists"}), 409
    t = {
        "id": tid,
        "dest":   data.get("dest",   "").strip().upper(),
        "status": data.get("status", "loaded"),
        "date":   data.get("date",   "").strip(),
        "notes":  data.get("notes",  "").strip(),
        "double": bool(data.get("double", False)),
        "slot":   data.get("slot") or None,
    }
    trailers.append(t)
    return jsonify(t), 201

@app.route("/api/trailers/<path:tid>", methods=["PUT"])
def update_trailer(tid):
    t = _find(tid)
    if not t:
        return jsonify({"error": "Not found"}), 404
    data = request.json
    for field in ("dest", "status", "date", "notes", "double", "slot"):
        if field in data:
            t[field] = data[field]
    return jsonify(t)

@app.route("/api/trailers/<path:tid>", methods=["DELETE"])
def delete_trailer(tid):
    global trailers
    t = _find(tid)
    if not t:
        return jsonify({"error": "Not found"}), 404
    trailers = [x for x in trailers if x["id"] != tid]
    return jsonify({"deleted": tid})

@app.route("/api/move", methods=["POST"])
def move_trailer():
    """Move a trailer to a slot (or unassign)."""
    data = request.json
    tid  = data.get("id")
    slot = data.get("slot") or None
    t = _find(tid)
    if not t:
        return jsonify({"error": "Not found"}), 404
    # Swap if target occupied
    if slot:
        occ = next((x for x in trailers if x["slot"] == slot), None)
        if occ:
            occ["slot"] = t["slot"]
    t["slot"] = slot
    return jsonify({"moved": tid, "slot": slot})

# ─────────────────────────────────────────────
# AI Assistant endpoint
# ─────────────────────────────────────────────
@app.route("/api/ai", methods=["POST"])
def ai_assist():
    data    = request.json or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])   # [{role, content}, …]

    if not message:
        return jsonify({"error": "No message"}), 400

    client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from env

    system_prompt = f"""You are YARD-AI, an expert operations assistant for Container Yard Planner — YARD CHECK.
You have real-time knowledge of the current yard state.

CURRENT YARD SNAPSHOT:
{_snapshot()}

You help yard managers with:
- Trailer status queries ("where is SPAN 401748?")
- Slot recommendations ("where should I put a loaded FBX trailer?")
- Workload analysis ("what's the detention situation?")
- Prioritisation advice ("which trailers need attention first?")
- Pattern recognition and anomaly detection
- Shift handover summaries

Be concise, direct, and operational. Use short bullet points. Always reference real trailer IDs and slot numbers from the snapshot.
If the user asks you to move or change something, explain what API call to make or describe the action clearly.
"""

    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )

    reply = response.content[0].text
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
