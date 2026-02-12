import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # This allows your HTML file to talk to this script

# Replace with your actual Gemini API Key
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/analyze-sound', methods=['POST'])
def analyze_sound():
    data = request.json
    word = data.get('word')
    # We take the average frequency values to send to the AI
    freq_values = data.get('frequencies') 
    
    prompt = f"""
    A student is learning STEM acoustics. They recorded the word '{word}'.
    The frequency intensity data (0-255 scale) for this sound is: {freq_values}.
    In 2 short sentences, explain a specific visual feature of this sound wave 
    (like if it's high-pitched/noisy or low/smooth) to help them recognize it later.
    """
    
    response = model.generate_content(prompt)
    return jsonify({"analysis": response.text})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))