import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Use a simple password of your choice
MY_SECRET_PASSWORD = "stem_explorer_2026"

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/analyze-sound', methods=['POST'])
def analyze_sound():
    # SECURITY CHECK: Look for the password in the headers
    client_password = request.headers.get("X-Custom-Password")
    
    if client_password != MY_SECRET_PASSWORD:
        return jsonify({"error": "Unauthorized Access"}), 403

    data = request.json
    word = data.get('word')
    freq_values = data.get('frequencies') 
    
    prompt = f"Analyze the STEM acoustic features of the word '{word}' with frequencies {freq_values}. Max 20 words."
    
    response = model.generate_content(prompt)
    return jsonify({"analysis": response.text})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)