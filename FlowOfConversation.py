from __future__ import division

import re
import sys
import os
import time
import playsound
import speech_recognition as sr 
from gtts import gTTS 


import dialogflow
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue
from google.api_core.exceptions import InvalidArgument

from googletrans import Translator
translatorFrench = Translator(service_urls=['translate.google.com', 'translate.google.fr',])
translatorSpanish = Translator(service_urls=['translate.google.com', 'translate.google.es',])
translatorDutch = Translator(service_urls=['translate.google.com', 'translate.google.de',])
translatorItalian = Translator(service_urls=['translate.google.com', 'translate.google.it',])


query = "poo"


#INFO PERTAINING MY DIALOGFLOW AGENT
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'private_key.json'

DIALOGFLOW_PROJECT_ID = 'tatiana-sdccby'
DIALOGFLOW_LANGUAGE_CODE = 'en'
SESSION_ID = 'me'

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

#Actual textual output converted from Audio Input
textualOutput = ""
text = ""

performSearch = "Google"
performTranslate = "Translate"
translate = "translate"
languageFrench = "French"
languageSpanish = "Spanish"
languageDutch = "Dutch"
languageItalian = "Italian"

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
	
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:

            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            textualOutput = (transcript + overwrite_chars)
            print ("Output is: " + textualOutput)

            if textualOutput == performSearch:
                print ("PERFORM GOOGLE SEARCH " + query)
                google(query)
                

            if textualOutput == translate or performTranslate:
                translateInput = textualOutput
                chooseLanguage(translateInput)
                
            
            ##NEED TO SEND DIFFERENT TEXT TO DIALOGFLOW THAN TO TRANSLATION
            dialogFlow(textualOutput)
            #print(transcript + overwrite_chars)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0



#DEALS WITH GOOGLE'S DIALOGFLOW. SENDS TEXTUAL VERSION OF USER'S AUDIO INPUT TO
#GOOGLE DIALOGFLOW IN ORDER TO PROMPT A MEANINGFUL RESPONSE
def dialogFlow(textual_request):
    print ("Dialogflow textual request is:" + textual_request)
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
    text_input = dialogflow.types.TextInput(text=textual_request, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)
    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
    except InvalidArgument:
        raise

    print("Query text:", response.query_result.query_text)
    print("Detected intent:", response.query_result.intent.display_name)
    print("Detected intent confidence:", response.query_result.intent_detection_confidence)
    print("Fulfillment text:", response.query_result.fulfillment_text)

    print ("Output from google is: " + response.query_result.fulfillment_text)
    text = response.query_result.fulfillment_text
    #print"Input to speech is: " + text_response_from_google
    speak(text)




def speak(text):
    tts = gTTS(text = text, lang = "en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)




def google(query):
    print (query)
    #response = GoogleSearch().search()
    #(query, tld='com', lang='en', num=10, start=0, stop=None, pause=2.0)
    response = GoogleSearch().search(query)
    for result in response.results:
        print("Title: " + result.title)
        print("Content: " + result.getText())


def translateIntoFrench(input):
    translations = translatorFrench.translate([input], dest = "fr")
    for translation in translations:
        print(translation.text)

    return translation.text

def translateIntoItalian(input):
    translations = translatorItalian.translate([input], dest = "it")
    for translation in translations:
        print(translation.text)

    return translation.text

def translateIntoDutch(input):
    translations = translatorDutch.translate([input], dest = "nl")
    for translation in translations:
        print(translation.text)

    return translation.text

def translateIntoSpanish(input):
    translations = translatorSpanish.translate([input], dest = "es")
    for translation in translations:
        print(translation.text)

    return translation.text

def chooseLanguage(textualOutput):

    if languageFrench in textualOutput:
        start = textualOutput.find("Translate") + len("Translate")
        end = textualOutput.find("French")
        substring = textualOutput[start+2:end]
        print (substring)
        print ("PERFORM TRANSLATION FRENCH")
        translatedText = translateIntoFrench(substring)
        speak(translatedText)
        return
                  

    if languageItalian in textualOutput:
        start = textualOutput.find("Translate") + len("Translate")
        end = textualOutput.find("Italian")
        substring = textualOutput[start+2:end]
        print (substring)
        print ("PERFORM TRANSLATION Italian")
        translatedText = translateIntoItalian(substring)
        speak(translatedText)
        return


    if languageDutch in textualOutput:
        start = textualOutput.find("Translate") + len("Translate")
        end = textualOutput.find("Dutch")
        substring = textualOutput[start+2:end]
        print (substring)
        print ("PERFORM TRANSLATION Dutch")
        translatedText = translateIntoDutch(substring)
        speak(translatedText)
        return
    

    if languageSpanish in textualOutput:
        start = textualOutput.find("Translate") + len("Translate")
        end = textualOutput.find("Spanish")
        substring = textualOutput[start+2:end]
        print (substring)
        print ("PERFORM TRANSLATION Spanish")
        translatedText = translateIntoSpanish(substring)
        speak(translatedText)
        return







def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'en-US'  # a BCP-47 language tag

    #STT CONFIGURATION
    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)
        # Now, put the transcription responses to use.
        print ("HELLO") 
        listen_print_loop(responses)






if __name__ == '__main__':
    main()