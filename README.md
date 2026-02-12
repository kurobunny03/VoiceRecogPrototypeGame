# VoiceRecogPrototypeGame

# 🧬 Sonic Fingerprint Lab: STEM Sound Explorer

An interactive web-based STEM game that teaches students about acoustics, frequency, and pattern recognition. Using the Web Audio API and Gemini AI, players visualize their own voices as waterfall plots (spectrograms) and challenge themselves to recognize "sonic fingerprints."

## 🚀 Live Demo
* **Game:** [Your GitHub Pages URL Here]
* **AI Backend:** [Your Render/Railway URL Here]

---

## 🔬 STEM Concepts Explored
* **Frequency vs. Time:** Understanding how sounds are mapped on a 2D plane (Waterfall Plot).
* **Acoustic Phonetics:** Identifying the difference between vowels (harmonics) and consonants (fricatives/noise).
* **Machine Learning:** Exploring how AI can analyze data patterns to assist human identification.



---

## 🎮 How to Play
1.  **Start Microphone:** Grant permissions to see the live waterfall plot.
2.  **Capture Dataset:** Type a word (e.g., "Burger"), say it, and hit **Capture**.
3.  **Build Your Lab:** Collect at least 4 different words. The AI will provide a "Lab Report" for each sound.
4.  **Enter Challenge Mode:** Identify which of your previously recorded waveforms is the "Mystery Pattern."

---

## 🛠️ Tech Stack
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Web Audio API).
* **Backend:** Python (Flask), CORS.
* **AI:** Gemini 1.5 Flash API (via Google Generative AI SDK).
* **Deployment:** GitHub Pages (Frontend) + Render/Railway (Backend).

---

## ⚙️ Setup & Installation

### 1. Backend (Python)
1. Navigate to the root folder.
2. Create a virtual environment: `python -m venv venv`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Set your Environment Variable: `export GEMINI_API_KEY='your_key_here'`.
5. Run the server: `python app.py`.

### 2. Frontend
1. Open `index.html` in any modern browser.
2. Ensure the `fetch` URL in `index.html` matches your backend address (`http://localhost:5000` or your live URL).

---

## 🔒 Security Note
This project uses a **Flask Proxy Architecture** to protect API keys. The Gemini API key is stored as an environment variable on the server side and is never exposed to the client-side browser code.
