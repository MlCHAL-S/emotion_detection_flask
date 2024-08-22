"""
Emotion detection module
"""
import os
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


def emotion_detector(text_to_analyze):
    """Analyzes the emotion of the given text using Watson NLU API.

    Returns a dictionary with emotion scores and the dominant emotion,
    or None if the input is invalid.
    """
    api_key = os.environ.get('WATSON_API_KEY')
    service_url = os.environ.get('WATSON_URL')

    if not api_key or not service_url:
        raise ValueError("API key and service URL must be set in environment variables.")

    authenticator = IAMAuthenticator(api_key)
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2022-04-07',
        authenticator=authenticator
    )
    natural_language_understanding.set_service_url(service_url)

    if not text_to_analyze.strip():
        # Return None for all values if the input text is empty or whitespace
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    response = natural_language_understanding.analyze(
        text=text_to_analyze,
        features=Features(emotion=EmotionOptions())
    ).get_result()

    # Extracting emotion labels and scores from the response
    emotions = response['emotion']['document']['emotion']
    dominant_emotion = max(emotions, key=emotions.get)

    # Returning a dictionary containing emotion analysis results
    return {
        'anger': emotions.get('anger'),
        'disgust': emotions.get('disgust'),
        'fear': emotions.get('fear'),
        'joy': emotions.get('joy'),
        'sadness': emotions.get('sadness'),
        'dominant_emotion': dominant_emotion
    }
