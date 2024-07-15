import streamlit as st
import openai
from gtts import gTTS
import pyaudio
import wave
import numpy as np
import os
import tempfile
import requests
import threading
import time

# Set up API keys from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

def generate_response_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_response_groq(prompt):
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

def speech_to_text(audio_file):
    with open(audio_file, "rb") as file:
        transcript = openai.Audio.transcribe("whisper-1", file)
    return transcript["text"]

def record_audio(filename):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

st.title("AI Chatbot with Simulated Live Speech")

api_choice = st.radio("Choose API:", ("OpenAI", "GROQ"))

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.button("Start Conversation"):
    user_input = ""
    while True:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            record_audio(tmp_file.name)
            user_input = speech_to_text(tmp_file.name)
            os.unlink(tmp_file.name)

        if user_input.lower() in ["stop", "quit", "exit"]:
            break

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if api_choice == "OpenAI":
            response = generate_response_openai(user_input)
        else:
            response = generate_response_groq(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        audio_file = text_to_speech(response)
        st.audio(audio_file)
        os.unlink(audio_file)

        time.sleep(1)  # Small delay to prevent rapid-fire API calls

    st.write("Conversation ended.")

st.write("Say 'stop', 'quit', or 'exit' to end the conversation.")
