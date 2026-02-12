import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# 1. SETUP FLASK
app = Flask(__name__)
CORS(app) # Allow all origins for now

# 2. SETUP SECURITY & API
MY_SECRET_PASSWORD = "stem_explorer_2026"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# 3. ROUTES
@app.route('/', methods=['GET'])
def health_check():
    return "STEM Lab is Online!", 200

@app.route('/analyze-sound', methods=['POST'])
def analyze_sound():
    try:
        # SECURITY CHECK
        client_password = request.headers.get("X-Custom-Password")
        if client_password != MY_SECRET_PASSWORD:
            print(f"Auth Failed: Received {client_password}")
            return jsonify({"error": "Unauthorized Access"}), 403

        # PARSE DATA
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        word = data.get('word', 'Unknown')
        freq_values = data.get('frequencies', [])
        
        # LOGGING (Check Render Logs for this!)
        print(f"Analyzing word: {word}") 
        
        # ASK GEMINI
        prompt = f"""
        Act as a STEM Physics Tutor.
        Student said: '{word}'
        Frequency Data (0-255 scale): {freq_values}
        Task: Explain the visual shape of this sound in 1-2 sentences. 
        Focus on pitch (low/high) and texture (smooth/noisy).
        """
        
        response = model.generate_content(prompt)
        
        if not response.text:
            return jsonify({"error": "Empty response from AI"}), 500
            
        return jsonify({"analysis": response.text})

    except Exception as e:
        # CRITICAL DEBUGGING: Print exact error to Render Logs
        print("CRITICAL ERROR IN ANALYZE_SOUND:")
        print(traceback.format_exc()) 
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)