import streamlit as st
import openai
from gtts import gTTS
import os
import tempfile
import requests  # We'll use this for GROQ API calls

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

st.title("AI Chatbot with Text-to-Speech")

api_choice = st.radio("Choose API:", ("OpenAI", "GROQ"))

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        # Generate response based on API choice
        if api_choice == "OpenAI":
            response = generate_response_openai(user_input)
        else:
            response = generate_response_groq(user_input)
        
        st.text_area("AI Response:", value=response, height=120)
        
        # Convert response to speech
        audio_file = text_to_speech(response)
        
        # Play the audio
        st.audio(audio_file)
        
        # Clean up the temporary audio file
        os.unlink(audio_file)
