!pip install SpeechRecognition gtts gradio
import gradio as gr
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import webbrowser
import threading

# --------------------------
# Setup
# --------------------------
LOG_FILE = "session_log.csv"

# Dummy therapy data
THERAPY_DATA = {
    "articulation": "Practice tongue twisters focusing on 'r' and 's' sounds.",
    "fluency": "Try slow breathing exercises before speaking.",
    "language": "Practice forming simple sentences and gradually increase complexity.",
    "voice": "Do humming exercises to strengthen vocal cords."
}

# Training a simple classifier for text-based therapy suggestion
texts = [
    "I can't pronounce r sounds",
    "I speak very fast and stutter",
    "I have trouble forming sentences",
    "My voice is very weak"
]
labels = ["articulation", "fluency", "language", "voice"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

# --------------------------
# Utility functions
# --------------------------
def log_session(feature, content):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[now, feature, content]], columns=["Time", "Feature", "Content"])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def show_dashboard():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        counts = df['Feature'].value_counts()
        plt.figure(figsize=(6,4))
        counts.plot(kind='bar')
        plt.title("Feature Usage Dashboard")
        plt.ylabel("Count")
        plt.xlabel("Feature")
        plt.tight_layout()
        plt.savefig("dashboard.png")
        return "dashboard.png"
    return None

# Therapy Suggestion

def therapy_suggestion(user_text):
    X_test = vectorizer.transform([user_text])
    pred = model.predict(X_test)[0]
    suggestion = THERAPY_DATA.get(pred, "Try general speech practice.")
    log_session("Therapy", f"Input: {user_text} | Predicted: {pred}")
    return f"Predicted Area: {pred}\nSuggestion: {suggestion}"

# Speech-to-Text

def speech_to_text(audio):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            log_session("Speech-to-Text", text)
            return text
        except:
            return "Could not recognize speech."

# Text-to-Speech

def text_to_speech(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        log_session("Text-to-Speech", text)
        return fp.name

# --------------------------
# Gradio Interface
# --------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🗣️ Speech Therapy & Communication Assistant")

    with gr.Tab("Dashboard"):
        dash_btn = gr.Button("Show Dashboard")
        dash_img = gr.Image()
        dash_btn.click(fn=show_dashboard, outputs=dash_img)

    with gr.Tab("Therapy Suggestions"):
        user_in = gr.Textbox(label="Describe your difficulty")
        suggestion_out = gr.Textbox(label="Therapy Suggestion")
        submit_btn = gr.Button("Get Suggestion")
        submit_btn.click(fn=therapy_suggestion, inputs=user_in, outputs=suggestion_out)

    with gr.Tab("Speech-to-Text"):
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath")
        text_out = gr.Textbox()
        audio_btn = gr.Button("Transcribe")
        audio_btn.click(fn=speech_to_text, inputs=audio_in, outputs=text_out)

    with gr.Tab("Text-to-Speech"):
        tts_in = gr.Textbox(label="Enter text")
        tts_out = gr.Audio()
        tts_btn = gr.Button("Convert to Speech")
        tts_btn.click(fn=text_to_speech, inputs=tts_in, outputs=tts_out)

# --------------------------
# Launch in Colab or Local
# --------------------------
import sys
is_colab = 'google.colab' in sys.modules

def launch_and_open():
    if is_colab:
        demo.launch(share=True)
    else:
        url = demo.launch()
        if isinstance(url, str):
            threading.Timer(2, lambda: webbrowser.open(url)).start()

if __name__ == "__main__":
    launch_and_open()
