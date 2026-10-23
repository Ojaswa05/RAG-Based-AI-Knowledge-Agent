"""RAG Teaching Assistant - chat backend.
Run from the folder that contains embeddings.joblib (and optionally Videos/):
    python app.py   ->  http://localhost:5000
"""
import os, re, json, requests, numpy as np, joblib
from flask import Flask, request, Response, jsonify, send_from_directory
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
DATA = os.environ.get("EMBEDDINGS", "embeddings.joblib")
VIDEO_DIR = os.environ.get("VIDEO_DIR", "Videos")

app = Flask(__name__, static_folder="static", static_url_path="")
df = joblib.load(DATA)
EMB = np.vstack(df["embedding"].values)


def video_map():
    """Map lecture number -> file name inside Videos/ (e.g. 'Lec 07 - Title.mp4')."""
    out = {}
    if os.path.isdir(VIDEO_DIR):
        for f in os.listdir(VIDEO_DIR):
            if f.lower().endswith((".mp4", ".webm", ".mkv")):
                m = re.search(r"(\d+)", f)
                if m:
                    out[int(m.group(1))] = f
    return out


VIDEOS = video_map()


def num(n):
    m = re.search(r"\d+", str(n))
    return int(m.group()) if m else None


def embed(text, model):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": model, "input": [text], "keep_alive": "30m"}, timeout=120)
    return r.json()["embeddings"][0]


def retrieve(query, k, model):
    sims = cosine_similarity(EMB, [embed(query, model)]).flatten()
    idx = sims.argsort()[::-1][:k]
    out = []
    for i in idx:
        row = df.iloc[i]
        out.append({
            "title": row["title"], "number": str(row["number"]), "start": float(row["start"]),
            "end": float(row["end"]), "text": row["text"].strip(), "score": float(sims[i]),
            "video": num(row["number"]) in VIDEOS,
        })
    return sorted(out, key=lambda c: (num(c["number"]) or 0, c["start"]))


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/models")
def models():
    try:
        tags = requests.get(f"{OLLAMA}/api/tags", timeout=5).json().get("models", [])
        names = [m["name"] for m in tags if "embed" not in m["name"] and "bge" not in m["name"]]
        return jsonify({"models": names, "ollama": True, "chunks": len(df), "videos": len(VIDEOS)})
    except Exception:
        return jsonify({"models": [], "ollama": False, "chunks": len(df), "videos": len(VIDEOS)})


@app.get("/video/<int:n>")
def video(n):
    if n not in VIDEOS:
        return "No video file for this lecture", 404
    return send_from_directory(VIDEO_DIR, VIDEOS[n], conditional=True)


@app.post("/api/chat")
def chat():
    d = request.get_json()
    msgs = [{"role": m["role"], "content": m["content"]} for m in d["messages"] if m.get("content")]
    k, temp = int(d.get("top_k", 5)), float(d.get("temperature", 0.4))
    model, rag = d.get("model", "llama3.2"), d.get("rag", True)
    system = d.get("system", "")

    def gen():
        try:
            sources = []
            system_full = system
            if rag:
                users = [m["content"] for m in msgs if m["role"] == "user"][-2:]
                sources = retrieve(" ".join(users), k, EMBED_MODEL)
                notes = "\n".join(
                    f"- Video {s['number']} ({s['title']}), {int(s['start'])}s to {int(s['end'])}s: {s['text']}"
                    for s in sources)
                q = msgs[-1]["content"]
                # put context + question in the LAST user turn (works much better for small models)
                msgs[-1] = {"role": "user", "content": (
                    "Below are notes from the course videos. They are only for your reference.\n\n"
                    f"{notes}\n\n"
                    f"Student question: {q}\n\n"
                    "Now reply like a helpful teacher in plain, natural, conversational language "
                    "(in the same language as the question). Explain in your own words which video "
                    "teaches this and at what timestamp (mm:ss), and tell the student where to go. "
                    "Do NOT copy or list the notes, do NOT output JSON, bullets of raw chunks or the word 'chunk'. "
                    "If the question is unrelated to the course, say you can only answer course-related questions.")}
            yield json.dumps({"sources": sources}) + "\n"
            body = {"model": model, "stream": True, "keep_alive": "30m",
            "options": {"temperature": temp, "num_predict": int(d.get("max_tokens", 350)), "num_ctx": 3072},
                    "messages": [{"role": "system", "content": system_full}] + msgs}
            with requests.post(f"{OLLAMA}/api/chat", json=body, stream=True, timeout=600) as r:
                for line in r.iter_lines():
                    if line:
                        j = json.loads(line)
                        if "error" in j:
                            yield json.dumps({"error": j["error"]}) + "\n"
                            return
                        yield json.dumps({"t": j.get("message", {}).get("content", "")}) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as e:
            yield json.dumps({"error": f"{type(e).__name__}: {e}. Is Ollama running?"}) + "\n"

    return Response(gen(), mimetype="application/x-ndjson")


def warmup():
    """Load both models into memory at startup so the first question is not slow."""
    try:
        embed("warm up", EMBED_MODEL)
        requests.post(f"{OLLAMA}/api/generate", json={"model": "llama3.2", "prompt": "hi", "stream": False,
                      "keep_alive": "30m", "options": {"num_predict": 1}}, timeout=300)
    except Exception:
        pass


if __name__ == "__main__":
    import threading
    threading.Thread(target=warmup, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
