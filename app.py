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

st.title("AI Chatbot with Simulated Speech Input")

api_choice = st.radio("Choose API:", ("OpenAI", "GROQ"))

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.write("Enter your message in the text box below and click 'Speak' to simulate speech input. Type 'stop', 'quit', or 'exit' to end the conversation.")

user_input = st.text_input("Your message:", "")

if st.button("Speak"):
    if user_input:
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
