import streamlit as st
import openai
from gtts import gTTS
import os
import tempfile

# Set up OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

st.title("ChatGPT with Text-to-Speech")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        # Generate response
        response = generate_response(user_input)
        st.text_area("ChatGPT:", value=response, height=120)
        
        # Convert response to speech
        audio_file = text_to_speech(response)
        
        # Play the audio
        st.audio(audio_file)
        
        # Clean up the temporary audio file
        os.unlink(audio_file)
