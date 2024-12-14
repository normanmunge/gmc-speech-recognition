import streamlit as st
import speech_recognition as sr
import wave
import librosa
import numpy as np
import pyaudio
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Load the dataset
with open('data.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Define function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Function to find the most relevant sentence in a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0 
    most_relevant_sentence = ''
    
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = ' '.join(sentence)
            
    return most_relevant_sentence

def chatbot(question):
    # Find the most relevant sentence in the text
    most_relevant_sentence = get_most_relevant_sentence(question)
    
    # Return the answer
    return most_relevant_sentence

# Function to transcribe speech using Google Speech Recognition
def transcribe_speech():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            st.info("Speak now...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.info("Transcribing...")
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand what you said."
    except sr.RequestError:
        return "Request error. Please check your internet connection."
    except Exception as e:
        return f"An error occurred: {e}"

# Main function
def main():
    st.title("Chatbot with Speech Recognition")
    st.write("Hello, I'm Butangi - the M.H.S Alumni chatbot. How can I help you today?")
    
    # Input options: Text or Voice
    input_method = st.radio("How would you like to ask your question?", ("Type", "Speak"))

    question = ""
    if input_method == "Type":
        question = st.text_input("Type your question:")
    elif input_method == "Speak":
        if st.button("Start Speaking"):
            question = transcribe_speech()
            st.write(f"You said: {question}")

    # Create a button to submit the question
    if st.button("Ask"):
        if question.strip():
            response = chatbot(question)
            st.write(f'Chatbot: {response}')
        else:
            st.write("Please provide a valid question.")

if __name__ == '__main__':
    main()
