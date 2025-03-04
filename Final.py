import pyaudio
import json
import os 
import requests
from google.cloud import speech
import logging
import pygame
import time
import numpy as np
from io import BytesIO
from google.cloud import texttospeech, speech
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load credentials for Text-to-Speech and Speech-to-Text
try:
    tts_credentials = service_account.Credentials.from_service_account_file(
        '/Users/carolistical/Desktop/multi_modal/service_account_tts.json'
    )
    stt_credentials = service_account.Credentials.from_service_account_file(
        '/Users/carolistical/Desktop/multi_modal/service_account_stt.json'
    )
except Exception as e:
    logger.error(f"Error loading credentials: {e}")
    raise

# Create clients with specific credentials
tts_client = texttospeech.TextToSpeechClient(credentials=tts_credentials)
stt_client = speech.SpeechClient(credentials=stt_credentials)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Configure audio stream with larger buffer and more robust settings
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=2048,  # Increased buffer size
    stream_callback=None
)
stream.start_stream()

# Speech recognition configuration
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US"
)

# Streaming recognition configuration
streaming_config = speech.StreamingRecognitionConfig(
    config=config,
    interim_results=True,
    single_utterance=False
)

def generate_audio_requests(stream, chunk_size=1024):
    """
    Enhanced audio request generator with more sophisticated timeout handling
    """
    try:
        silence_threshold = 3.0  # Increased silence threshold
        max_stream_duration = 120  # Increased max stream duration
        
        start_time = time.time()
        last_audio_time = start_time
        total_silence_duration = 0

        while True:
            try:
                # Read larger audio chunk
                audio_data = stream.read(chunk_size, exception_on_overflow=False)
                
                current_time = time.time()
                
                # Check audio energy to determine if it's meaningful sound
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array**2))
                
                if rms > 50:  # Adjust this threshold based on your environment
                    last_audio_time = current_time
                    total_silence_duration = 0
                    yield speech.StreamingRecognizeRequest(audio_content=audio_data)
                else:
                    # Track silence duration
                    total_silence_duration += (current_time - last_audio_time)
                    
                    # Optional: Send silent chunks to keep connection alive
                    yield speech.StreamingRecognizeRequest(audio_content=audio_data)
                
                # Check for excessive silence
                if total_silence_duration > silence_threshold:
                    logger.warning(f"Extended silence detected: {total_silence_duration:.2f} seconds")
                    break
                
                # Check total stream duration
                if (current_time - start_time) > max_stream_duration:
                    logger.warning(f"Maximum stream duration reached: {max_stream_duration} seconds")
                    break

            except IOError as io_error:
                logger.error(f"IO Error in audio stream: {io_error}")
                break

    except Exception as e:
        logger.error(f"Unexpected error in audio generation: {e}")
        raise

# Initialize Pygame mixer for playing audio
pygame.mixer.init()

def speak_and_transcribe(text, language="en-GB"):
    try:
        stream.stop_stream()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
            name="en-GB-Chirp-HD-F",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            effects_profile_id=["small-bluetooth-speaker-class-device"]
        )
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        # Play speech
        audio_buffer = BytesIO(response.audio_content)
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait for the speech to finish playing
            time.sleep(0.1)
        stream.start_stream()
    
    except Exception as e:
        logger.error(f"Error in text-to-speech synthesis: {e}")

# OpenRouter API configuration
API_KEY = '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'
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_openrouter_api(user_input, conversation_history, API_URL, API_KEY):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Append user input to conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Prepare the data to send to OpenRouter API
    data = json.dumps({
        "model": "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        "messages": [
            {
                "role": "system",
                "content": """
                    You are a real estate agent in the UAE, helping the user find a property based on their preferences
                    by asking clear,concise and professional questions. Keep the conversation friendly and straightforward, without
                    any unecessary reasoning. Ask one question at a time and wait for the customer's response before moving on to the
                    next question. Speak clearly and supress any random noises. 
                    These are the questions you need to ask in the following order:
                    1. What is the number of bedrooms you are looking for?
                    2. What is the wanted location of the property?
                    3. Can you provide a budget range in AED?
                    4. What is the maximum property age you are considering in years?
                """
            },
            *conversation_history,  # Include only the conversation history
        ],
        "temperature": 0.4,
        "include_reasoning": False
    })
    
    try:
        response = requests.post(API_URL, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        
        response_data = response.json()
        
        if 'choices' in response_data and response_data['choices']:
            assistant_response = response_data["choices"][0]["message"]["content"]
            
            # Handle empty content gracefully
            if not assistant_response:
                assistant_response = "Sorry, I didn't get that. Could you please clarify or rephrase?"

            # Append the assistant's response to conversation history
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
            print("Assistant Response: " + assistant_response)
            speak_and_transcribe(assistant_response)
            return assistant_response
        else:
            assistant_response = "Sorry, could you repeat that for me?"
            conversation_history.append({"role": "assistant", "content": assistant_response})
            speak_and_transcribe(assistant_response)
            return assistant_response
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OpenRouter API: {e}")
        assistant_response = "Failed to get a response from the API. Please try again."
        conversation_history.append({"role": "assistant", "content": assistant_response})
        speak_and_transcribe(assistant_response)
        return assistant_response

def start_speech_recognition(stream, stt_client, streaming_config):
    """
    Enhanced speech recognition with error handling and reset mechanism
    """
    conversation_history = []
    
    while True:
        try:
            # Reset audio requests generator each time
            audio_requests = generate_audio_requests(stream)
            
            # Start streaming recognition
            responses = stt_client.streaming_recognize(streaming_config, audio_requests)
            
            # Process responses
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        final_transcript = result.alternatives[0].transcript
                        print(f"Recognized Speech: {final_transcript}")
                        
                        # Call OpenRouter API with the recognized speech
                        try:
                            call_openrouter_api(final_transcript, conversation_history, API_URL, API_KEY)
                        except Exception as api_error:
                            logger.error(f"API call error: {api_error}")
        
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            time.sleep(1)  # Prevent the rapid error looping

def main():
    try:
        logger.info("Starting speech recognition...")
        start_speech_recognition(stream, stt_client, streaming_config)
    except KeyboardInterrupt:
        logger.info("Stopping speech recognition...")
    finally:
        # General clean up
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
