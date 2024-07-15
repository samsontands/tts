import streamlit as st
import openai
from gtts import gTTS
import os
import tempfile
import requests
import time

# Set up API keys from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

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

st.title("AI Chatbot with Simulated Live Speech")

api_choice = st.radio("Choose API:", ("OpenAI", "GROQ"))

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.write("Click 'Start Recording' and speak. Click 'Stop Recording' when you're done. Say 'stop', 'quit', or 'exit' to end the conversation.")

audio = st.audio_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording")

if audio is not None:
    # Save the recorded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio.tobytes())
        tmp_file_path = tmp_file.name

    # Transcribe the audio
    user_input = speech_to_text(tmp_file_path)
    os.unlink(tmp_file_path)

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if user_input.lower() in ["stop", "quit", "exit"]:
        st.write("Conversation ended.")
    else:
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

    st.experimental_rerun()
