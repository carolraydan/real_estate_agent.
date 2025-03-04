import os
from google.cloud import texttospeech
import speech_recognition as sr
import pygame.mixer
import time

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_account.json'

# Initialize Google Cloud TTS client
client = texttospeech.TextToSpeechClient()

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize pygame mixer
pygame.mixer.init()

# Simulated knowledge base (general UAE property info with English translations only)
property_info = {
    "Dubai": {
        "under_500k_en": "Apartments and smaller townhouses in areas like Jumeirah Village Circle or Dubai Silicon Oasis."
    },
    "Abu Dhabi": {
        "under_500k_en": "Apartments in areas like Al Reem Island or Khalifa City."
    }
}

# Function to speak and transcribe response using Google TTS (English only)
def speak_and_transcribe(text, language="en"):
    print(f"Transcription: {text}")  # Simulate text output on screen
    
    # Configure voice for English only
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-GB",  # English language code
        name="en-GB-Chirp-HD-F",  # Your chosen British female voice
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Audio configuration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        effects_profile_id=["small-bluetooth-speaker-class-device"]
    )

    # Synthesize speech
    synthesis_input = texttospeech.SynthesisInput(text=text)
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Save and play audio using pygame
    audio_file = "output.mp3"
    with open(audio_file, "wb") as output:
        output.write(response.audio_content)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for playback to finish
        time.sleep(0.1)
    os.remove(audio_file)  # Clean up temporary file

# Function to detect language from user input (always English for this use case)
def detect_language(text):
    return "en"

# Function to process property inquiries
def process_inquiry(query, language="en"):
    query = query.lower()
    if language == "en":
        response = "Hello! I’m Emma, your property assistant. "

    # Parse query and provide property details (only in English)
    if "dubai" in query and "under $500,000" in query:
        response += property_info["Dubai"]["under_500k_en"] + " Prices vary by location and features. Could you specify if you want a flat or something else?"
    elif "abu dhabi" in query and "under $500,000" in query:
        response += property_info["Abu Dhabi"]["under_500k_en"] + " Let me know if you’d like more details!"
    else:
        response += "I can help with properties across the UAE. Could you tell me which city and budget you’re interested in?"

    speak_and_transcribe(response, language)

# Main function to handle voice input
def main():
    print("Starting Emma, your UAE Property Assistant...")
    speak_and_transcribe("Hello! I’m Emma, your property assistant. How may I assist you with UAE properties today?", "en")

    while True:
        try:
            with sr.Microphone() as source:
                print("Listening for your question...")
                audio = recognizer.listen(source, timeout=5)
                query = recognizer.recognize_google(audio)
                print(f"You said: {query}")

                # Detect language (always English for this version)
                language = detect_language(query)

                # Process the inquiry
                process_inquiry(query, language)

        except sr.UnknownValueError:
            speak_and_transcribe("Sorry, I didn’t catch that. Could you please repeat your question?", language)
        except sr.RequestError:
            speak_and_transcribe("I’m having trouble connecting. Please try again.", language)
        except KeyboardInterrupt:
            speak_and_transcribe("Goodbye! Thanks for using Emma.", language)
            break

if __name__ == "__main__":
    main()
