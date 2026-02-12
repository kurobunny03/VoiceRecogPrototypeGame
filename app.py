import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


app = Flask(__name__)

CORS(app, resources={
    r"/analyze-sound": {"origins": [
        "https://voicerecogprototypegame.onrender.com",
        "http://localhost:5500"
    ]}
}, allow_headers=["Content-Type", "X-Custom-Password"])


limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])

MY_SECRET_PASSWORD = os.environ.get("MY_SECRET_PASSWORD")

# 1. SETUP OPENAI CLIENT
# We use the standard environment variable name
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

@app.route('/', methods=['GET'])
def health_check():
    return "STEM Lab (OpenAI Edition) is Online!", 200

@app.route('/analyze-sound', methods=['POST'])
@limiter.limit("20 per minute")
def analyze_sound():
    try:
        # Security Check
        client_password = request.headers.get("X-Custom-Password")
        if client_password != MY_SECRET_PASSWORD:
            print(f"Auth Failed.")
            return jsonify({"error": "Unauthorized Access"}), 403

        data = request.json
        word = data.get('word', 'Unknown')
        freq_values = data.get('frequencies', [])
        
        print(f"Analyzing word: {word}...")

        # 2. THE OPENAI PROMPT
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Fast, cheap, and smart
            messages=[
                {"role": "system", "content": "You are a Physics Tutor. Analyze the sound data provided."},
                {"role": "user", "content": f"""
                The student said the word: '{word}'
                Here is the frequency data (0-255 scale) from the Fast Fourier Transform:
                {freq_values}
                
                Task: Explain the visual shape of this sound in 1-2 short sentences. 
                Focus on pitch (low vs high energy) and texture.
                """}
            ]
        )

        # 3. EXTRACT ANSWER
        ai_response = completion.choices[0].message.content
        return jsonify({"analysis": ai_response})

    except Exception as e:
        print("CRITICAL ERROR:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)