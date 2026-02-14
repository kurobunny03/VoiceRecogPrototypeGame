import os
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

MY_SECRET_PASSWORD = os.environ.get("MY_SECRET_PASSWORD", "stem_explorer_2026")
ENABLE_TTS = os.environ.get("ENABLE_TTS", "0") == "1"

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

@app.route('/', methods=['GET'])
def health_check():
    return "STEM Lab (OpenAI Edition) is Online!", 200

def require_password(req):
    client_password = req.headers.get("X-Custom-Password")
    return client_password == MY_SECRET_PASSWORD

@app.route('/analyze-sound', methods=['POST'])
def analyze_sound():
    try:
        if not require_password(request):
            return jsonify({"error": "Unauthorized Access"}), 403

        data = request.get_json(silent=True) or {}
        word = data.get('word', 'Unknown')
        freq_values = data.get('frequencies', [])

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Physics and Signal Processing Tutor. Analyze FFT data quantitatively and explain results clearly."
                },
                {
                    "role": "user",
                    "content": f"""
The student said the word: '{word}'

Here is the FFT frequency magnitude data (0–255 scale):
{freq_values}

Perform the following analysis:
1. Identify the index of the highest amplitude value (dominant frequency bin).
2. Estimate whether the dominant energy is in the lower 33%, middle 33%, or upper 33% of frequencies.
3. Estimate the average amplitude of the dataset.
4. Briefly describe what this implies about the pitch (low, mid, high).

Respond in this structured format:
- Dominant Bin Index: ___
- Dominant Region: ___
- Estimated Average Amplitude: ___
- Interpretation (1-2 sentences): ___
"""
                }
            ]
        )

        ai_response = completion.choices[0].message.content
        return jsonify({"analysis": ai_response})

    except Exception:
        print("CRITICAL ERROR:")
        print(traceback.format_exc())
        return jsonify({"error": "Server error"}), 500


# ============================
# NEW: AUDIO DIALOG ENDPOINT
# ============================
@app.route('/dialog', methods=['POST'])
def dialog():
    """
    multipart/form-data:
      - audio: webm/ogg blob from MediaRecorder
      - context: JSON string with:
          {
            "mode": "challenge" or "lab",
            "currentWord": "...",
            "targetWord": "...",             (optional)
            "fft": [0..255] (optional),
            "analysisText": "...",           (optional)
            "difficulty": 1..5,
            "points": int,
            "history": [{"role":"user|assistant","content":"..."}]
          }
    returns JSON:
      {
        "transcript": "...",
        "reply": "...",
        "score": 0..1,
        "pointsEarned": int,
        "totalPoints": int,
        "difficulty": int,
        "nextQuestion": "...",
        "ttsAudioBase64": "..." (optional)
      }
    """
    try:
        if not require_password(request):
            return jsonify({"error": "Unauthorized Access"}), 403

        if "audio" not in request.files:
            return jsonify({"error": "Missing audio file"}), 400

        ctx_raw = request.form.get("context", "{}")
        try:
            ctx = json.loads(ctx_raw)
        except Exception:
            ctx = {}

        audio_file = request.files["audio"]

        # 1) TRANSCRIBE (STT)
        # Note: OpenAI python SDK uses client.audio.transcriptions.create
        transcript_obj = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=(audio_file.filename, audio_file.stream, audio_file.mimetype),
        )
        transcript = transcript_obj.text.strip() if transcript_obj and transcript_obj.text else ""

        # 2) BUILD A RUBRIC + DIFFICULTY LADDER
        difficulty = int(ctx.get("difficulty", 1))
        total_points = int(ctx.get("points", 0))

        fft = ctx.get("fft", None)
        analysis_text = ctx.get("analysisText", "")

        history = ctx.get("history", [])
        if not isinstance(history, list):
            history = []

        system = """You are an interactive Physics/Signal Processing tutor for students.
You are conducting an oral quiz about sound waves and FFT/spectrogram patterns.

You must:
- Respond conversationally and encouragingly (briefly).
- Grade the student's answer with a score from 0.0 to 1.0.
- Award points: 0..10 based on score and difficulty.
- Increase difficulty when the student does well; decrease or keep when struggling.
- Provide ONE next question appropriate to the new difficulty level.

Return STRICT JSON only, using this schema:
{
  "reply": string,
  "score": number,
  "pointsEarned": integer,
  "newDifficulty": integer,
  "nextQuestion": string
}
"""

        # Provide context to the model
        ctx_summary = {
            "difficulty": difficulty,
            "totalPoints": total_points,
            "currentWord": ctx.get("currentWord"),
            "targetWord": ctx.get("targetWord"),
            "analysisText": analysis_text,
            "fftPreview": fft[:40] if isinstance(fft, list) else None
        }

        messages = [{"role": "system", "content": system}]
        # include a bit of conversation history so it feels like a dialog
        for turn in history[-6:]:
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                messages.append({"role": turn["role"], "content": str(turn["content"])[:1500]})

        messages.append({
            "role": "user",
            "content": f"""Tutor context: {json.dumps(ctx_summary)}

Student said (transcript): {transcript}

If FFT preview is present, you may reference it qualitatively (low vs high energy, spikes/harmonics, spread/noise).
Question difficulty scale:
1=very easy conceptual (loud vs quiet)
2=pitch region (low/mid/high)
3=harmonics/texture and why
4=compare two spectra and justify
5=apply concept to a new scenario / explain reasoning clearly

Now respond and grade."""
        })

        # 3) GPT RESPONSE
        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4
        )

        raw = comp.choices[0].message.content.strip()

        # try to parse strict JSON
        try:
            payload = json.loads(raw)
        except Exception:
            # fallback: wrap it
            payload = {
                "reply": raw,
                "score": 0.5,
                "pointsEarned": 3,
                "newDifficulty": difficulty,
                "nextQuestion": "What do you think changes in the FFT when you speak louder?"
            }

        score = float(payload.get("score", 0.5))
        points_earned = int(payload.get("pointsEarned", 0))
        new_difficulty = int(payload.get("newDifficulty", difficulty))
        next_q = str(payload.get("nextQuestion", ""))[:400]
        reply = str(payload.get("reply", ""))[:1200]

        total_points += points_earned

        resp = {
            "transcript": transcript,
            "reply": reply,
            "score": max(0.0, min(1.0, score)),
            "pointsEarned": points_earned,
            "totalPoints": total_points,
            "difficulty": max(1, min(5, new_difficulty)),
            "nextQuestion": next_q
        }

        # 4) OPTIONAL TTS (AI voice reply)
        if ENABLE_TTS and reply:
            speech = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=f"{reply} Next question: {next_q}"
            )
            # speech is bytes-like; base64 it to send in JSON
            import base64
            audio_b64 = base64.b64encode(speech.read()).decode("utf-8")
            resp["ttsAudioBase64"] = audio_b64
            resp["ttsMime"] = "audio/mpeg"

        return jsonify(resp)

    except Exception:
        print("CRITICAL ERROR:")
        print(traceback.format_exc())
        return jsonify({"error": "Server error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
