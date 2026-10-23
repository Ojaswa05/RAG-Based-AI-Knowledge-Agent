import os
import json
import whisper

# Agar "jsons" folder exist nahi karta toh create karein
os.makedirs("jsons", exist_ok=True)

# Whisper model load karein
model = whisper.load_model("large-v2")

audios = os.listdir("audios")

# Files ko number ke hisaab se sort karne ke liye helper function
def get_file_number(filename):
    if "." in filename:
        num_part = filename.split(".", 1)[0].strip()
        if num_part.isdigit():
            return int(num_part)
    return float("inf")

# Files ko 1, 2, ..., 9, 10 ke numeric order me sort karein
sorted_audios = sorted(audios, key=get_file_number)

for audio in sorted_audios:
    if audio.endswith(".mp3") and "." in audio:
        parts = audio.split(".", 1)
        number_str = parts[0].strip()

        # Check karein ki audio number 10 ya usse bada ho
        if number_str.isdigit() and int(number_str) >= 1:
            title = parts[1].replace(".mp3", "").strip()

            # Agar JSON file pehle se bani hui ho toh skip kar dega
            json_filename = audio.replace(".mp3", ".json")
            json_path = os.path.join("jsons", json_filename)
            if os.path.exists(json_path):
                print(f"Skipping already processed: {audio}")
                continue

            print(f"Processing ({number_str}): {title}")

            result = model.transcribe(
                audio=f"audios/{audio}",
                language="hi",
                task="translate",
                word_timestamps=False
            )

            chunks = []
            for segment in result["segments"]:
                chunks.append({
                    "number": number_str,
                    "title": title,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                })

            chunks_with_metadata = {
                "chunks": chunks,
                "text": result["text"]
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)