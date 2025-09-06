# -*- coding: utf-8 -*-
"""
NLP-Enhanced Communication for Speech Disorders
------------------------------------------------
Single-file demo app with:
- ASR (simulated / real if installed)
- NLP disorder analysis
- Structured therapy suggestions
- Text-to-Speech (gTTS preferred, pyttsx3 fallback)
- Logging + simple dashboard
- Gradio UI (Dashboard, Therapy, ASR, TTS)
"""
!pip install SpeechRecognition gTTS pyttsx3 gradio
# -----------------------------------------------------------------------------
# Imports
import os
import sys
import json
import logging
import tempfile
import threading
import webbrowser
from datetime import datetime
import speech_recognition as sr
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Optional dependencies (best-effort detection)
try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except Exception:
    HAS_PYTTSX3 = False

try:
    import speech_recognition as sr
    HAS_SR = True
except Exception:
    HAS_SR = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
LOG_FILE = "session_log.csv"

def now_iso():
    return datetime.now().isoformat()

def log_session(feature: str, content: str):
    """Append a line to CSV log for simple dashboard analytics."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[now, feature, content]], columns=["Time", "Feature", "Content"])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def show_dashboard():
    """Return a PNG path showing counts per feature (if logs exist)."""
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        counts = df['Feature'].value_counts()
        plt.figure(figsize=(6,4))
        counts.plot(kind='bar')
        plt.title("Feature Usage Dashboard")
        plt.ylabel("Count")
        plt.xlabel("Feature")
        plt.tight_layout()
        out = "dashboard.png"
        plt.savefig(out)
        plt.close()
        return out
    return None

# -----------------------------------------------------------------------------
# Robust TTS: try gTTS (mp3) then pyttsx3 (wav) fallback
# -----------------------------------------------------------------------------
def safe_save_tts(text: str):
    """
    Try to generate an audio file and return its filepath.
    Uses gTTS (mp3) if available; otherwise falls back to pyttsx3 (wav).
    Returns (filepath, status_message).
    """
    if not text or not text.strip():
        return None, "Empty text provided."

    # First try gTTS (mp3)
    if HAS_GTTS:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.close()
            filename = tmp.name
            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            return filename, f"gTTS saved MP3 → {os.path.basename(filename)}"
        except Exception as e:
            logging.warning(f"gTTS generation failed: {e}")

    # Fallback to pyttsx3 (WAV)
    if HAS_PYTTSX3:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            filename = tmp.name
            engine = pyttsx3.init()
            # Optional: tune voice rate/volume if desired
            # engine.setProperty('rate', 150)
            engine.save_to_file(text, filename)
            engine.runAndWait()
            return filename, f"pyttsx3 saved WAV → {os.path.basename(filename)}"
        except Exception as e:
            logging.error(f"pyttsx3 generation failed: {e}")

    # Nothing available
    return None, "No TTS engine available. Install 'gTTS' (pip install gTTS) or 'pyttsx3' (pip install pyttsx3)."

# -----------------------------------------------------------------------------
# Core ASR/NLP/Recommendation logic (kept simple for demo)
# -----------------------------------------------------------------------------
class ASRWrapper:
    def __init__(self, simulate=True):
        self.simulate = simulate
        # For production, plug in Whisper/transformers pipeline here

    def transcribe(self, audio_file: str):
        if self.simulate:
            # A simulated, "stuttered" example
            return {"text": "I.. I w-want to go to the p-park", "confidence": 0.85}
        # If you want to enable real ASR, implement here (e.g., use transformers)
        return {"text": "", "confidence": 0.0}

class NLPAnalyzer:
    def analyze(self, transcript: str):
        if not transcript:
            return {"type": "Unknown", "severity": 0.0, "patterns": []}
        tokens = transcript.split()
        repetition = any(tokens.count(tok) > 2 for tok in set(tokens))
        if ".." in transcript or "-" in transcript or repetition:
            return {"type": "Stuttering", "severity": 0.8, "patterns": ["Repetition", "Prolongation"]}
        if len(transcript) / max(1, len(tokens)) > 7:
            return {"type": "Apraxia", "severity": 0.7, "patterns": ["Phoneme issues"]}
        return {"type": "Neurological", "severity": 0.5, "patterns": ["Vague Speech"]}

class RecommendationEngine:
    RULES = {
        "stuttering": [
            "Practice slow, relaxed breathing.",
            "Pacing strategies — pause between words.",
            "Try rhythm/metronome-based practice."
        ],
        "apraxia": [
            "Break words into syllables and practice slowly.",
            "Mirror-assisted articulation practice.",
            "Repetition with visual cues."
        ],
        "neurological": [
            "Use phrase banks and AAC supports.",
            "Practice common daily phrases repeatedly.",
            "Leverage visual aids/communication boards."
        ],
        "unknown": [
            "General speech warm-ups daily.",
            "Consult a licensed speech-language pathologist."
        ]
    }
    def recommend(self, speech_type: str):
        return self.RULES.get(speech_type.lower(), self.RULES["unknown"])

# -----------------------------------------------------------------------------
# Therapy classifier (tiny demo model)
# -----------------------------------------------------------------------------
THERAPY_DATA = {
    "articulation": {
        "focus": "Improve clarity of specific speech sounds.",
        "exercises": [
            "Tongue twisters for 'r' and 's'.",
            "Mirror exercises to observe mouth shape.",
            "Record-and-playback to self-correct."
        ],
        "tools": ["Mirror", "Articulation flashcards", "Recording app"]
    },
    "fluency": {
        "focus": "Increase smoothness and reduce stuttering.",
        "exercises": [
            "Slow breathing before speaking.",
            "Paced reading with a metronome.",
            "Short phrase practice building to longer phrases."
        ],
        "tools": ["Metronome app", "Paced reading apps"]
    },
    "language": {
        "focus": "Improve sentence formation and comprehension.",
        "exercises": [
            "Form simple SVO sentences.",
            "Question-answer practice.",
            "Story retelling in short steps."
        ],
        "tools": ["Picture cards", "Story-building apps"]
    },
    "voice": {
        "focus": "Strengthen vocal fold function and projection.",
        "exercises": [
            "Humming and glides (low to high).",
            "Sustained phonation practice.",
            "Controlled reading with pauses."
        ],
        "tools": ["Pitch tracker", "Voice exercises"]
    }
}

_demo_texts = [
    "I can't pronounce r sounds",
    "I speak very fast and stutter",
    "I have trouble forming sentences",
    "My voice is very weak"
]
_demo_labels = ["articulation", "fluency", "language", "voice"]
_vectorizer = TfidfVectorizer()
_X = _vectorizer.fit_transform(_demo_texts)
_therapy_model = LogisticRegression()
_therapy_model.fit(_X, _demo_labels)

def therapy_suggestion(user_text: str):
    if not user_text or not user_text.strip():
        return "Please enter a description of the difficulty."
    x_test = _vectorizer.transform([user_text])
    pred = _therapy_model.predict(x_test)[0]
    details = THERAPY_DATA.get(pred, {"focus":"General practice","exercises":[],"tools":[]})
    # Format output nicely for Gradio Markdown or Textbox
    out_lines = [
        f"**Predicted area:** {pred}",
        f"**Focus:** {details.get('focus','-')}",
        "**Exercises:**"
    ]
    for ex in details.get("exercises", []):
        out_lines.append(f"- {ex}")
    out_lines.append("**Tools:**")
    for t in details.get("tools", []):
        out_lines.append(f"- {t}")
    log_session("Therapy", f"Input: {user_text} | Predicted: {pred}")
    return "\n".join(out_lines)

# -----------------------------------------------------------------------------
# Gradio function wrappers
# -----------------------------------------------------------------------------
def speech_to_text_file(audio):
    """If speech_recognition is available, transcribe; otherwise return simulated message."""
    if not audio:
        return "No audio file provided."
    if not HAS_SR:
        return "SpeechRecognition not installed. Use ASR simulation instead."
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            log_session("Speech-to-Text", text)
            return text
    except Exception as e:
        logging.error(f"Speech-to-text error: {e}")
        return "Could not transcribe audio."

def text_to_speech_fn(text):
    """Gradio wrapper: create audio file and status message."""
    filepath, status = safe_save_tts(text)
    if filepath:
        log_session("Text-to-Speech", text)
        # Gradio Audio will accept the returned path (mp3 or wav)
        return filepath, status
    else:
        # Return None for audio + status message
        return None, status

def run_asr_analysis(audio):
    asr = ASRWrapper(simulate=True)
    out = asr.transcribe(audio)
    transcript = out.get("text","")
    conf = out.get("confidence", 0.0)
    analysis = NLPAnalyzer().analyze(transcript)
    recs = RecommendationEngine().recommend(analysis.get("type","unknown"))

    analysis_str = (
        f"Type: {analysis.get('type')}\n"
        f"Severity: {analysis.get('severity')}\n"
        f"Patterns: {', '.join(analysis.get('patterns', []))}\n"
        f"ASR confidence: {conf:.2f}"
    )
    recs_str = "\n".join([f"- {r}" for r in recs])
    log_session("ASR", transcript)
    return transcript, analysis_str, recs_str

# -----------------------------------------------------------------------------
# Build Gradio app
# -----------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🗣️ NLP-Enhanced Communication for Speech Disorders (Demo)")

    with gr.Tab("📊 Dashboard"):
        gr.Markdown("Usage log (click to generate a simple chart):")
        dash_btn = gr.Button("Show Dashboard")
        dash_img = gr.Image()
        dash_btn.click(fn=show_dashboard, outputs=dash_img)

    with gr.Tab("🧠 Therapy Suggestions"):
        gr.Markdown("Describe the user's difficulty and get a structured suggestion.")
        user_in = gr.Textbox(label="Describe difficulty", placeholder="e.g. I can't say the 'r' sound clearly")
        suggestion_out = gr.Markdown()
        submit_btn = gr.Button("Get Suggestion")
        submit_btn.click(fn=therapy_suggestion, inputs=user_in, outputs=suggestion_out)

    with gr.Tab("🎙️ Speech-to-Text"):
        gr.Markdown("Upload audio or record from microphone (local runtime).")
        audio_in = gr.Audio(type="filepath")
        transcribed = gr.Textbox(label="Transcription")
        transcribe_btn = gr.Button("Transcribe (if SpeechRecognition available)")
        transcribe_btn.click(fn=speech_to_text_file, inputs=audio_in, outputs=transcribed)

    with gr.Tab("🔊 Text-to-Speech"):
        gr.Markdown("Enter text and generate audio. If gTTS works you'll get an MP3; otherwise WAV via pyttsx3.")
        tts_in = gr.Textbox(label="Enter text to speak", lines=4, placeholder="Hello — I will speak this text.")
        tts_out = gr.Audio(label="Generated Speech")
        tts_status = gr.Textbox(label="Status / Notes")
        tts_btn = gr.Button("Convert to Speech")
        tts_btn.click(fn=text_to_speech_fn, inputs=tts_in, outputs=[tts_out, tts_status])

    with gr.Tab("🤖 ASR + NLP Analysis"):
        gr.Markdown("Run the ASR (simulated) then analyze for likely disorder types and recommendations.")
        asr_audio = gr.Audio(type="filepath")
        transcript_box = gr.Textbox(label="Transcript")
        analysis_box = gr.Textbox(label="Analysis")
        recs_box = gr.Textbox(label="Recommendations")
        asr_btn = gr.Button("Run ASR & Analysis")
        asr_btn.click(fn=run_asr_analysis, inputs=asr_audio, outputs=[transcript_box, analysis_box, recs_box])

# -----------------------------------------------------------------------------
# Launch helper
# -----------------------------------------------------------------------------
def launch_and_open():
    is_colab = 'google.colab' in sys.modules
    if is_colab:
        demo.launch(share=True)
    else:
        url = demo.launch()
        if isinstance(url, str):
            # open in browser
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()

if __name__ == "__main__":
    launch_and_open()
