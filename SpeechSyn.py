import azure.cognitiveservices.speech as speechsdk

def synthesize_ssml(ssml):
    speech_key = ""
    service_region = "eastus"
    # Create a speech configuration object with the key and region
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Create a speech synthesizer object
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    # Synthesize the SSML and play it as audio
    result = synthesizer.speak_text_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Audio was successfully synthesized and played
        print("Speech synthesized successfully")
    elif result.reason == speechsdk.ResultReason.Canceled:
        # Speech synthesis was canceled
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

# Replace with your own subscription key and region
speech_key = ""
service_region = "eastus"
