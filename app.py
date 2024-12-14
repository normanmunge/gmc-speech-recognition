import streamlit as st
import speech_recognition as sr
import wave
import librosa
import numpy as np
import pyaudio
        
def transcribe_google(audio, language="en-US"):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        try:
            text = recognizer.recognize_google(audio, language=language)
            return text
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
        except sr.UnknownValueError as e:
            return "Sorry! I didn't get that!!! Please try again!"

def transcribe_pyaudio(audio):
    try:
        # Audio recording parameters
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        rate = 44100  # Record at 44100 samples per second
        duration = 5  # Record for 5 seconds
        output_filename = "output.wav"

        audio = pyaudio.PyAudio()
        stream = audio.open(format=sample_format, channels=channels,
                            rate=rate, input=True, frames_per_buffer=chunk)
        
        frames = []

        # Record for the given duration
        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded audio to a file
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(sample_format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        # Use SpeechRecognition for transcription
        recognizer = sr.Recognizer()
        with sr.AudioFile(output_filename) as source:
            audio_data = recognizer.record(source)
            return transcribe_google(audio_data)
    except Exception as e:
        return f"PyAudio Error: {e}"

def transcribe_deepgram(audio, api_key):
    import requests
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    try:
        response = requests.post(url, headers=headers, data=audio.get_wav_data())
        if response.status_code == 200:
            return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
        else:
            return f"Deepgram API Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Deepgram API Error: {e}"

def transcribe_librosa(audio):
    try:
        uploaded_file = st.file_uploader("Upload an audio file for Librosa analysis:", type=["wav", "mp3"])
        if uploaded_file is not None:
            y, sr = librosa.load(uploaded_file, sr=None)
            st.info("Audio file loaded successfully.")

            # Display waveform statistics
            duration = librosa.get_duration(y=y, sr=sr)
            st.write(f"**Duration:** {duration:.2f} seconds")
            st.write(f"**Sample Rate:** {sr} Hz")

            return "Librosa doesn't support transcription directly, but audio analysis is complete."
        else:
            return "Please upload an audio file for analysis."
    except Exception as e:
        return f"Librosa Error: {e}"

    
    

def transcribe_speech(api_choice, language="en-US", deepgram_api_key=None):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Listening... Speak very clearly into the microphone.")

        try:
            audio = recognizer.listen(source, timeout=30)
            st.info("Processing your input...")
            text = recognizer.recognize_google(audio, language=language)

            if api_choice == 'Google':
                text = transcribe_google(audio, language=language)
            elif api_choice == 'PyAudio':
                text = transcribe_pyaudio(audio)
            elif api_choice == 'Deepgram':
                text = transcribe_deepgram(audio, deepgram_api_key)
            elif api_choice == 'Librosa':
                text = transcribe_librosa(audio)
            else:
                text = "API choice not implemented yet."

            return text
        except sr.RequestError as e:
            return e
        except sr.UnknownValueError as e:
            return e
        except sr.WaitTimeoutError as e:
            return "Sorry! You took too long to respond. Please try again!"
        except Exception as e:
            return f"An unexpected error occurred: {e}"


    

def main():
    st.title("Speech Recognition App")
    st.subheader("Select the API and start speaking.")

    # API selection
    api_choice = st.radio("Select API for Transcription:", 
                          options=["Google", "PyAudio", "Deepgram", "Librosa"])

    # Language selection
    language = st.selectbox("Select Language for Transcription:", 
                             options=[("English", "en-US"), ("French", "fr-FR"), ("Spanish", "es-ES")],
                             format_func=lambda x: x[0])[1]
    
     # Input for Deepgram API key
    deepgram_key = None

    if api_choice == "Deepgram":
        deepgram_key = st.text_input("Enter your Deepgram API Key:", type="password")

    if st.button("Start Speaking"):
        if api_choice == "Deepgram" and not deepgram_key:
            st.error("Please provide your Deepgram API Key.")
        else:
            st.info("Microphone activated.")
            text = transcribe_speech(api_choice, language=language, deepgram_api_key=deepgram_key)
            st.write("**Transcription:**")
            st.text_area("Your Transcription:", text, height=100)

            # Save the transcription to a file
            if text and text != "Please upload an audio file for analysis.":
                file_name = "transcription.txt"
                st.download_button(
                    label="Download Transcription",
                    data=text,
                    file_name=file_name,
                    mime="text/plain")

if __name__ == '__main__':
    main()