import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
# Allow CORS for all domains (simplest for your setup)
CORS(app)

@app.route('/')
def health_check():
    return "STEM Lab is Online!", 200

# SECURITY: The password must match what is in your index.html
MY_SECRET_PASSWORD = "stem_explorer_2026"

# Configure Gemini
# Ensure you have set GEMINI_API_KEY in Render's Environment Variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# --- HEALTH CHECK ROUTE (CRITICAL FOR RENDER) ---
@app.route('/', methods=['GET'])
def home():
    return "STEM Audio Lab API is Online", 200

@app.route('/analyze-sound', methods=['POST'])
def analyze_sound():
    # 1. SECURITY CHECK
    client_password = request.headers.get("X-Custom-Password")
    if client_password != MY_SECRET_PASSWORD:
        return jsonify({"error": "Unauthorized Access"}), 403

    # 2. PARSE DATA
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    word = data.get('word', 'Unknown')
    freq_values = data.get('frequencies', [])

    # 3. ASK GEMINI
    prompt = f"""
    Act as a STEM Physics Tutor.
    Student said: '{word}'
    Frequency Data (0-255 scale): {freq_values}
    
    Task: Explain the visual shape of this sound in 1-2 sentences. 
    Focus on pitch (low/high) and texture (smooth/noisy).
    """
    
    try:
        response = model.generate_content(prompt)
        return jsonify({"analysis": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Render requires binding to 0.0.0.0 and the specifically assigned PORT
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)