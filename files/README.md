# Course Assistant (RAG chat app)

## Setup
1. Keep `app.py`, `index.html`, and `embeddings.joblib` in the same folder
   (and add a `Videos/` folder if you want in-app playback with timestamps).
2. `pip install -r requirements.txt`
3. Start Ollama and make sure the models exist: `ollama pull bge-m3` and `ollama pull llama3.2`
4. `python app.py` then open http://localhost:5000

Optional env vars: `OLLAMA_URL`, `EMBED_MODEL`, `EMBEDDINGS`, `VIDEO_DIR`.
Video files are matched to lectures by the first number in the file name (e.g. `Lec 07 - Title.mp4` -> lecture 7).
