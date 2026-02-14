# VoiceRecogPrototypeGame

# 🧬 Sonic Fingerprint Lab: STEM Sound Explorer

An interactive web-based STEM game that teaches students about acoustics, frequency, and pattern recognition. Using the Web Audio API and OpenAI, players visualize their own voices as waterfall plots (spectrograms) and challenge themselves to recognize "sonic fingerprints."


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
* **AI:** OPENAI API (gpt-4o-mini)
* **Deployment:** GitHub Pages (Frontend) + Render (https://voicerecogprototypegame.onrender.com)


## 🔒 Security Note
This project uses a **Flask Proxy Architecture** to protect API keys. The API key is stored as an environment variable on the server side and is never exposed to the client-side browser code.
